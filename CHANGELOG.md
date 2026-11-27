# Changelog

All notable changes to muninn are recorded here, written for people who use muninn rather than people who read its source.

## Unreleased

- Project started, under the MIT license.
- The foundation is in place: a shared file format with corruption detection, exact (brute-force) vector search with cosine, dot-product and Euclidean similarity, and filtering of results by document.
- Exact vector search is much faster: it uses every CPU core and applies filters while it searches, so a filtered search costs no more than an unfiltered one.
- Fast approximate vector search (HNSW) for large collections. It builds on all CPU cores, accepts new vectors at any time (also after being reopened from disk), and on our test data finds 99% or more of the true ten nearest neighbours at its default settings.
- Filtered vector search always returns as many results as you asked for whenever that many documents match the filter, and picks the fastest way to do it automatically, including when only a handful of documents match.
- Optional vector compression (one byte or one bit per dimension) to search with less memory traffic; the best candidates are re-checked against the original vectors, so accuracy stays high.
- Vector indexes are saved as a single file and open instantly, however large they are: nothing has to be read into memory up front.
