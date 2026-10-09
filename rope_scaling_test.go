package llama_test

import (
	"math"
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/tcpipuk/llama-go"
)

// RoPE scaling test suite
//
// Tests the RoPE/YaRN context options, covering:
// - Rejection of invalid scaling methods and out-of-range values
// - Acceptance of every option with valid values
// - Overrides reaching llama.cpp, shown by a change in the embeddings

var _ = Describe("RoPE scaling options", func() {
	var (
		model     *llama.Model
		modelPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_EMBEDDING_MODEL")
		if modelPath == "" {
			Skip("TEST_EMBEDDING_MODEL not set - skipping integration test")
		}

		var err error
		model, err = llama.LoadModel(modelPath, llama.WithGPULayers(-1))
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if model != nil {
			Expect(model.Close()).To(Succeed())
		}
	})

	DescribeTable("should reject invalid values",
		func(opt llama.ContextOption, message string) {
			ctx, err := model.NewContext(llama.WithEmbeddings(), opt)
			Expect(err).To(MatchError(ContainSubstring(message)))
			Expect(ctx).To(BeNil())
		},
		Entry("unknown scaling method", llama.WithRopeScaling("yarm"), "invalid RoPE scaling method"),
		Entry("empty scaling method", llama.WithRopeScaling(""), "invalid RoPE scaling method"),
		Entry("zero frequency base", llama.WithRopeFreqBase(0), "invalid RoPE frequency base"),
		Entry("negative frequency scale", llama.WithRopeFreqScale(-0.5), "invalid RoPE frequency scale"),
		Entry("NaN frequency scale", llama.WithRopeFreqScale(float32(math.NaN())), "invalid RoPE frequency scale"),
		Entry("extrapolation factor above 1", llama.WithYarnExtFactor(1.5), "invalid YaRN extrapolation factor"),
		Entry("zero attention factor", llama.WithYarnAttnFactor(0), "invalid YaRN attention factor"),
		Entry("negative beta_fast", llama.WithYarnBetaFast(-1), "invalid YaRN beta_fast"),
		Entry("zero beta_slow", llama.WithYarnBetaSlow(0), "invalid YaRN beta_slow"),
		Entry("zero original context", llama.WithYarnOrigCtx(0), "invalid YaRN original context"),
		Label("integration"),
	)

	It("should report the first invalid option", Label("integration"), func() {
		_, err := model.NewContext(
			llama.WithEmbeddings(),
			llama.WithRopeScaling("bogus"),
			llama.WithRopeFreqScale(-1),
		)
		Expect(err).To(MatchError(ContainSubstring("invalid RoPE scaling method")))
	})

	It("should accept every option with valid values", Label("integration"), func() {
		ctx, err := model.NewContext(
			llama.WithEmbeddings(),
			llama.WithContext(4096),
			llama.WithRopeScaling("yarn"),
			llama.WithRopeFreqBase(1000000),
			llama.WithRopeFreqScale(0.5),
			llama.WithYarnExtFactor(1),
			llama.WithYarnAttnFactor(1),
			llama.WithYarnBetaFast(32),
			llama.WithYarnBetaSlow(1),
			llama.WithYarnOrigCtx(2048),
		)
		Expect(err).NotTo(HaveOccurred())
		defer func() { Expect(ctx.Close()).To(Succeed()) }()

		embeddings, err := ctx.GetEmbeddings("Hello world")
		Expect(err).NotTo(HaveOccurred())
		Expect(embeddings).NotTo(BeEmpty())
	})

	It("should change embeddings when scaling is applied", Label("integration"), func() {
		const text = "The quick brown fox jumps over the lazy dog near the riverbank."

		embed := func(opts ...llama.ContextOption) []float32 {
			ctx, err := model.NewContext(append([]llama.ContextOption{llama.WithEmbeddings()}, opts...)...)
			Expect(err).NotTo(HaveOccurred())
			defer func() { Expect(ctx.Close()).To(Succeed()) }()

			embeddings, err := ctx.GetEmbeddings(text)
			Expect(err).NotTo(HaveOccurred())
			return embeddings
		}

		baseline := embed()
		repeat := embed()
		scaled := embed(llama.WithRopeScaling("yarn"), llama.WithRopeFreqScale(0.75))
		Expect(scaled).To(HaveLen(len(baseline)))

		// GPU kernels need not be bit-exact between runs, so compare how far a
		// scaled run moves against how far a plain repeat drifts
		noise := maxAbsDiff(baseline, repeat)
		shift := maxAbsDiff(baseline, scaled)
		Expect(shift).To(BeNumerically(">", 10*noise+1e-4),
			"RoPE overrides should reach llama.cpp (shift %g, run-to-run noise %g)", shift, noise)
	})
})

func maxAbsDiff(a, b []float32) float64 {
	var maxDiff float64
	for i := range a {
		maxDiff = math.Max(maxDiff, math.Abs(float64(a[i]-b[i])))
	}
	return maxDiff
}
