//go:build shared_lib
// +build shared_lib

// Shared-mode CGO link flags.
//
// Selected by building with: go build -tags shared_lib (and producing the
// workspace with: BUILD_LINKAGE=shared make libbinding.a).
//
// In this mode the Makefile copies .so files into the workspace root and the
// resulting binary depends on them at runtime. The -Wl,-rpath,$ORIGIN flag
// bakes a relative-to-binary search path into the binary so the .so files only
// need to sit alongside it (no LD_LIBRARY_PATH required).

package llama

/*
#cgo LDFLAGS: -L./ -Wl,-rpath,$ORIGIN -lbinding -lllama-common -lllama -lggml -lggml-cpu -lggml-base -lstdc++ -lm -lgomp
*/
import "C"
