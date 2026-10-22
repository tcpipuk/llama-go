package llama_test

import (
	"context"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	llama "github.com/tcpipuk/llama-go"
)

var _ = Describe("Reasoning", func() {
	var model *llama.Model
	var ctx *llama.Context

	messages := []llama.ChatMessage{
		{Role: "user", Content: "What is 2+2? Answer in one word."},
	}

	BeforeEach(func() {
		// Qwen3 emits <think> blocks, so the chat model doubles as the reasoning model
		testModelPath := os.Getenv("TEST_CHAT_MODEL")
		if testModelPath == "" {
			Skip("TEST_CHAT_MODEL environment variable not set")
		}

		var err error
		model, err = llama.LoadModel(testModelPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())

		ctx, err = model.NewContext(llama.WithContext(4096))
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if ctx != nil {
			Expect(ctx.Close()).To(Succeed())
		}
		if model != nil {
			Expect(model.Close()).To(Succeed())
		}
	})

	Describe("Parsing output", Label("integration", "chat", "reasoning"), func() {
		output := "<think>\nTwo plus two.\n</think>\n\nFour"

		It("should extract reasoning with ReasoningFormatAuto", func() {
			content, reasoning, err := llama.ParseChatOutput(model, messages,
				llama.ChatOptions{ReasoningFormat: llama.ReasoningFormatAuto}, output)
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(reasoning)).To(Equal("Two plus two."))
			Expect(strings.TrimSpace(content)).To(Equal("Four"))
		})

		It("should leave reasoning inline with ReasoningFormatNone", func() {
			content, reasoning, err := llama.ParseChatOutput(model, messages,
				llama.ChatOptions{ReasoningFormat: llama.ReasoningFormatNone}, output)
			Expect(err).NotTo(HaveOccurred())
			Expect(reasoning).To(BeEmpty())
			Expect(content).To(ContainSubstring("<think>"))
			Expect(content).To(ContainSubstring("Four"))
		})

		It("should give the same result when fed in pieces", func() {
			content, reasoning, err := llama.ParseChatOutput(model, messages,
				llama.ChatOptions{ReasoningFormat: llama.ReasoningFormatAuto},
				"<thi", "nk>\nTwo plus", " two.\n</th", "ink>\n\nFo", "ur")
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(reasoning)).To(Equal("Two plus two."))
			Expect(strings.TrimSpace(content)).To(Equal("Four"))
		})
	})

	Describe("Rendering the prompt", Label("integration", "chat", "reasoning"), func() {
		It("should render the template's empty thinking block when thinking is off", func() {
			prompt, err := model.FormatChatPrompt(messages, llama.ChatOptions{EnableThinking: llama.Bool(false)})
			Expect(err).NotTo(HaveOccurred())
			Expect(prompt).To(ContainSubstring("</think>"))
		})

		It("should leave the thinking block to the model when thinking is on", func() {
			prompt, err := model.FormatChatPrompt(messages, llama.ChatOptions{EnableThinking: llama.Bool(true)})
			Expect(err).NotTo(HaveOccurred())
			Expect(prompt).NotTo(ContainSubstring("</think>"))
		})
	})

	Describe("Chat", Label("integration", "chat", "reasoning"), func() {
		opts := func(thinking bool) llama.ChatOptions {
			return llama.ChatOptions{
				MaxTokens:       llama.Int(1024),
				Temperature:     llama.Float32(0),
				EnableThinking:  llama.Bool(thinking),
				ReasoningFormat: llama.ReasoningFormatAuto,
			}
		}

		It("should separate reasoning from content when thinking", func() {
			response, err := ctx.Chat(context.Background(), messages, opts(true))
			Expect(err).NotTo(HaveOccurred())
			Expect(response.ReasoningContent).NotTo(BeEmpty())
			Expect(response.Content).NotTo(ContainSubstring("<think>"))
		})

		It("should return no reasoning when thinking is off", func() {
			response, err := ctx.Chat(context.Background(), messages, opts(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(response.ReasoningContent).To(BeEmpty())
			Expect(response.Content).NotTo(BeEmpty())
		})

		It("should stream the same reasoning and content as Chat", func() {
			response, err := ctx.Chat(context.Background(), messages, opts(true))
			Expect(err).NotTo(HaveOccurred())

			deltas, errs := ctx.ChatStream(context.Background(), messages, opts(true))
			var content, reasoning strings.Builder
			for delta := range deltas {
				content.WriteString(delta.Content)
				reasoning.WriteString(delta.ReasoningContent)
			}
			Expect(<-errs).NotTo(HaveOccurred())

			Expect(reasoning.String()).To(Equal(response.ReasoningContent))
			Expect(content.String()).To(Equal(response.Content))
		})

		It("should cut thinking short at the reasoning budget", func() {
			budgeted := opts(true)
			budgeted.ReasoningBudget = llama.Int(16)

			response, err := ctx.Chat(context.Background(), messages, budgeted)
			Expect(err).NotTo(HaveOccurred())
			Expect(response.Content).NotTo(BeEmpty())

			tokens, err := ctx.Tokenize(response.ReasoningContent)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically("<=", 20))
		})
	})
})
