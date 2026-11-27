# Changelog

All notable changes to muninn are recorded here, written for people who use muninn rather than people who read its source.

## Unreleased

- Project started, under the MIT license.
- The foundation is in place: a shared file format with corruption detection, exact (brute-force) vector search with cosine, dot-product and Euclidean similarity, and filtering of results by document.
- Vector comparisons are now several times faster on modern Intel, AMD and ARM processors: muninn automatically uses the fastest instructions your processor offers, with no setup. Vectors can also be compared in half precision, which halves the memory they need. Setting `MUNINN_ISA=scalar` (or `avx2`, `avx512`, `neon`) limits which processor features are used, for troubleshooting and comparisons.
