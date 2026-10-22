package llama

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	gocontext "context"
	"fmt"
	"strings"
	"unsafe"
)

// Chat implementation for Context is in context.go
// This file contains shared types, options, and helpers

// chatSession renders the prompt with the model's Jinja chat template and
// parses generated output with the parser built from the same render, so
// reasoning extraction always matches the template that produced the prompt.
type chatSession struct {
	ptr unsafe.Pointer
}

// newChatSession applies the chat template to messages.
//
// The template priority is: opts.ChatTemplate > model's GGUF template > error.
// For raw completion without templates, use Generate() instead of Chat().
func newChatSession(m *Model, messages []ChatMessage, opts ChatOptions, format ReasoningFormat) (*chatSession, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("messages cannot be empty")
	}

	var templates unsafe.Pointer
	if opts.ChatTemplate != "" {
		cTemplate := C.CString(opts.ChatTemplate)
		defer C.free(unsafe.Pointer(cTemplate))
		templates = C.llama_wrapper_chat_templates_init(m.modelPtr, cTemplate)
		if templates == nil {
			return nil, fmt.Errorf("failed to load chat template: %s", C.GoString(C.llama_wrapper_last_error()))
		}
		// The session copies what it needs from the templates
		defer C.llama_wrapper_chat_templates_free(templates)
	} else {
		if m.ChatTemplate() == "" {
			return nil, fmt.Errorf("no chat template available: provide ChatOptions.ChatTemplate or use a model with embedded template (or use Generate() for raw completion)")
		}
		var err error
		if templates, err = m.getChatTemplates(); err != nil {
			return nil, err
		}
	}

	cRoles := make([]*C.char, len(messages))
	cContents := make([]*C.char, len(messages))
	for i, msg := range messages {
		cRoles[i] = C.CString(msg.Role)
		cContents[i] = C.CString(msg.Content)
	}
	defer func() {
		for i := range messages {
			C.free(unsafe.Pointer(cRoles[i]))
			C.free(unsafe.Pointer(cContents[i]))
		}
	}()

	enableThinking := C.int(-1)
	if opts.EnableThinking != nil {
		enableThinking = 0
		if *opts.EnableThinking {
			enableThinking = 1
		}
	}

	ptr := C.llama_wrapper_chat_session_new(
		templates,
		(**C.char)(unsafe.Pointer(&cRoles[0])),
		(**C.char)(unsafe.Pointer(&cContents[0])),
		C.int(len(messages)),
		C.llama_wrapper_reasoning_format(format),
		enableThinking,
	)
	if ptr == nil {
		return nil, fmt.Errorf("failed to apply chat template: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	return &chatSession{ptr: ptr}, nil
}

// prompt returns the rendered prompt.
func (s *chatSession) prompt() string {
	return C.GoString(C.llama_wrapper_chat_session_prompt(s.ptr))
}

// feed appends generated text and returns the content and reasoning parsed so
// far. Set final on the last call so incomplete markup is resolved.
func (s *chatSession) feed(text string, final bool) (content, reasoningContent string, err error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.llama_wrapper_chat_session_feed(s.ptr, cText, C.bool(final))
	if result == nil {
		return "", "", fmt.Errorf("failed to parse chat output: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	defer C.llama_wrapper_free_parsed_message(result)

	content = C.GoString(result.content)
	if result.reasoning_content != nil {
		reasoningContent = C.GoString(result.reasoning_content)
	}
	return content, reasoningContent, nil
}

// cPtr returns the session for the C generate params, or nil without one.
func (s *chatSession) cPtr() unsafe.Pointer {
	if s == nil {
		return nil
	}
	return s.ptr
}

func (s *chatSession) close() {
	C.llama_wrapper_chat_session_free(s.ptr)
	s.ptr = nil
}

// chatGenerateOptions builds generation options from chat options.
func chatGenerateOptions(session *chatSession, opts ChatOptions) []GenerateOption {
	// Use user-provided stop words (no defaults - template handles this)
	budget := -1
	if opts.ReasoningBudget != nil {
		budget = *opts.ReasoningBudget
	}
	genOpts := []GenerateOption{
		WithStopWords(opts.StopWords...),
		withChatSession(session, budget),
	}

	if opts.MaxTokens != nil {
		genOpts = append(genOpts, WithMaxTokens(*opts.MaxTokens))
	}
	if opts.Temperature != nil {
		genOpts = append(genOpts, WithTemperature(*opts.Temperature))
	}
	if opts.TopP != nil {
		genOpts = append(genOpts, WithTopP(*opts.TopP))
	}
	if opts.TopK != nil {
		genOpts = append(genOpts, WithTopK(*opts.TopK))
	}
	if opts.Seed != nil {
		genOpts = append(genOpts, WithSeed(*opts.Seed))
	}
	return genOpts
}

// chatWithContext implements non-streaming chat completion using a specific context.
//
// This is an internal helper called by Context.Chat().
func (m *Model) chatWithContext(ctx gocontext.Context, c *Context, messages []ChatMessage, opts ChatOptions) (*ChatResponse, error) {
	session, err := newChatSession(m, messages, opts, opts.ReasoningFormat)
	if err != nil {
		return nil, err
	}
	defer session.close()

	tokenCh, errCh := c.GenerateChannel(ctx, session.prompt(), chatGenerateOptions(session, opts)...)

	var output strings.Builder

Loop:
	for {
		select {
		case token, ok := <-tokenCh:
			if !ok {
				break Loop
			}
			output.WriteString(token)
		case err := <-errCh:
			if err != nil {
				return nil, err
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	fullOutput := output.String()
	content, reasoning, err := session.feed(fullOutput, true)
	if err != nil {
		// If parsing fails, return content as-is without reasoning extraction
		return &ChatResponse{Content: fullOutput}, nil
	}

	return &ChatResponse{
		Content:          content,
		ReasoningContent: reasoning,
	}, nil
}

// chatStreamWithContext implements streaming chat completion using a specific context.
//
// This is an internal helper called by Context.ChatStream().
func (m *Model) chatStreamWithContext(ctx gocontext.Context, c *Context, messages []ChatMessage, opts ChatOptions) (<-chan ChatDelta, <-chan error) {
	bufferSize := 256
	if opts.StreamBufferSize > 0 {
		bufferSize = opts.StreamBufferSize
	}

	deltaCh := make(chan ChatDelta, bufferSize)
	errCh := make(chan error, 1)

	go func() {
		defer close(deltaCh)
		defer close(errCh)

		sendErr := func(err error) {
			select {
			case errCh <- err:
			default:
			}
		}

		// DeepSeekLegacy leaves reasoning inline when streaming
		inline := opts.ReasoningFormat == ReasoningFormatDeepSeekLegacy

		session, err := newChatSession(m, messages, opts, opts.ReasoningFormat)
		if err != nil {
			sendErr(err)
			return
		}
		defer session.close()

		tokenCh, genErrCh := c.GenerateChannel(ctx, session.prompt(), chatGenerateOptions(session, opts)...)

		// Previous parsed state for delta computation
		var prevContent, prevReasoning string
		parseFailed := false

		send := func(delta ChatDelta) bool {
			select {
			case deltaCh <- delta:
				return true
			case <-ctx.Done():
				return false
			}
		}

		// emit parses the output so far and sends whatever is new since the
		// last parse. If parsing fails, it sends text as-is from then on.
		emit := func(text string, final bool) bool {
			if inline || parseFailed {
				if text == "" {
					return true
				}
				return send(ChatDelta{Content: text})
			}

			content, reasoning, err := session.feed(text, final)
			if err != nil {
				parseFailed = true
				return text == "" || send(ChatDelta{Content: text})
			}

			var delta ChatDelta
			if strings.HasPrefix(content, prevContent) {
				delta.Content = content[len(prevContent):]
				prevContent = content
			}
			if strings.HasPrefix(reasoning, prevReasoning) {
				delta.ReasoningContent = reasoning[len(prevReasoning):]
				prevReasoning = reasoning
			}
			if delta.Content == "" && delta.ReasoningContent == "" {
				return true
			}
			return send(delta)
		}

	Loop:
		for {
			select {
			case token, ok := <-tokenCh:
				if !ok {
					break Loop
				}
				if !emit(token, false) {
					return
				}
			case err := <-genErrCh:
				if err != nil {
					sendErr(err)
					return
				}
			case <-ctx.Done():
				return
			}
		}

		// Resolve anything held back as possibly incomplete markup
		emit("", true)
	}()

	return deltaCh, errCh
}

// Int returns a pointer to the given int value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    MaxTokens: llama.Int(100),  // Instead of &100
//	}
func Int(v int) *int {
	return &v
}

// Float32 returns a pointer to the given float32 value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    Temperature: llama.Float32(0.7),  // Instead of &0.7
//	}
func Float32(v float32) *float32 {
	return &v
}

// Bool returns a pointer to the given bool value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    EnableThinking: llama.Bool(true),  // Instead of &true
//	}
func Bool(v bool) *bool {
	return &v
}
