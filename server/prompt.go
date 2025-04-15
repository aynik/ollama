package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/models/mllama"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

var errTooManyImages = errors.New("vision model only supports a single image per message")

// chatPrompt accepts a list of messages and returns the prompt and images that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, truncate bool) (prompt string, images []llm.ImageData, tokenCount int, _ error) {
	var system []api.Message

	isMllama := checkMllamaModelFamily(m)

	var imageNumTokens int
	// TODO: Ideally we would compute this from the projector metadata but some pieces are implementation dependent
	if isMllama {
		// Our mllama implementation packs all of the embeddings into a single token
		imageNumTokens = 1
	} else {
		// Clip images are represented as 768 tokens, each an embedding
		imageNumTokens = 768
	}

	for j := range msgs {
		if msgs[j].Role == "system" {
			system = append(system, msgs[j])
		}
	}

	n := len(msgs) - 1
	var tokens []int

	if !truncate {
		n = 0
	} else {
		// in reverse, find all messages that fit into context window
		for i := n; i >= 0; i-- {
			if isMllama && len(msgs[i].Images) > 1 {
				return "", nil, 0, errTooManyImages
			}

			// always include the last message
			if i == n {
				continue
			}

			var b bytes.Buffer
			if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[i:]...), Tools: tools}); err != nil {
				return "", nil, 0, err
			}

			var err error
			tokens, err = tokenize(ctx, b.String())
			if err != nil {
				return "", nil, 0, err
			}

			ctxLen := len(tokens)
			if m.ProjectorPaths != nil {
				for _, m := range msgs[i:] {
					ctxLen += imageNumTokens * len(m.Images)
				}
			}

			if ctxLen > opts.NumCtx {
				slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[i:]))
				break
			} else {
				n = i
			}
		}
	}

	currMsgIdx := n

	for _, msg := range msgs[currMsgIdx:] {
		if isMllama && len(msg.Images) > 1 {
			return "", nil, 0, errTooManyImages
		}
	}

	for cnt, msg := range msgs[currMsgIdx:] {
		prefix := ""
		imgPrompt := ""
		prompt := msg.Content

		for _, i := range msg.Images {
			var imgData llm.ImageData

			if isMllama {
				if len(m.ProjectorPaths) == 0 {
					imgData = llm.ImageData{
						ID:   len(images),
						Data: i,
					}
				} else {
					data, opts, err := mllama.Preprocess(bytes.NewReader(i))
					if err != nil {
						return "", nil, 0, err
					}

					buf := new(bytes.Buffer)
					err = binary.Write(buf, binary.LittleEndian, data)
					if err != nil {
						return "", nil, 0, err
					}

					ar, ok := opts["aspectRatioIndex"].(int)
					if !ok {
						return "", nil, 0, fmt.Errorf("missing aspect ratio for image")
					}

					imgData = llm.ImageData{
						ID:            len(images),
						Data:          buf.Bytes(),
						AspectRatioID: ar,
					}
				}
				imgPrompt = "<|image|>"
			} else {
				imgData = llm.ImageData{
					ID:   len(images),
					Data: i,
				}
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}

			images = append(images, imgData)
		}
		msgs[currMsgIdx+cnt].Content = prefix + imgPrompt + prompt
	}

	var b bytes.Buffer
	if err := m.Template.Execute(&b, template.Values{Messages: append(system, msgs[currMsgIdx:]...), Tools: tools}); err != nil {
		return "", nil, 0, err
	}

	finalPrompt := b.String()

	var err error
	tokens, err = tokenize(ctx, finalPrompt)
	if err != nil {
		return "", nil, 0, err
	}

	tokenCount = len(tokens)

	if m.ProjectorPaths != nil {
		for _, m := range msgs[currMsgIdx:] {
			tokenCount += imageNumTokens * len(m.Images)
		}
	}

	return finalPrompt, images, tokenCount, nil
}

func checkMllamaModelFamily(m *Model) bool {
	for _, arch := range m.Config.ModelFamilies {
		if arch == "mllama" {
			return true
		}
	}
	return false
}
