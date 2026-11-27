# Changelog

All notable changes to muninn are recorded here, written for people who use muninn rather than people who read its source.

## Unreleased

- Project started, under the MIT license.
- The foundation is in place: a shared file format with corruption detection, exact (brute-force) vector search with cosine, dot-product and Euclidean similarity, and filtering of results by document.
- Full-text search: documents can be searched by keywords and ranked by relevance (BM25, the standard used by Elasticsearch and Lucene). A document matches if it contains any of the query words; documents containing rarer words, more often, in shorter text rank higher.
- Text is matched regardless of upper/lower case and works for any language that separates words with spaces, including accented and non-Latin scripts. Optionally, accents can be ignored ("café" finds "cafe"), common English words such as "the" and "and" can be skipped, and English word endings can be ignored ("running" finds "run"). The chosen settings are saved with the index, so searches always treat text the same way it was indexed.
- Newly added text is searchable immediately, and search stays fast as collections grow: only the documents that can still make the top results are examined, yet the results are exactly the same as checking every document.
- Search filters (for example "only documents from this user") are applied while searching, so you always get the requested number of results when enough matching documents exist.
- Relevance scores stay consistent across a whole collection even when it is stored in several parts, so results from different parts can be combined fairly.
- Text indexes are stored compactly on disk (about 2 bytes per word in our tests), open instantly, and are checked for corruption. Indexes can be combined into one, with deleted documents removed.
- A search may contain up to 1024 different words; longer searches are refused with a clear error. Repeating a word does not count against the limit.
- Current limitation: languages written without spaces between words (Chinese, Japanese, Thai) are not yet split into words, so searching them only finds exact runs of text.
- Vector comparisons are now several times faster on modern Intel, AMD and ARM processors: muninn automatically uses the fastest instructions your processor offers, with no setup. Vectors can also be compared in half precision, which halves the memory they need. Setting `MUNINN_ISA=scalar` (or `avx2`, `avx512`, `neon`) limits which processor features are used, for troubleshooting and comparisons.
- Index files with damaged trailing bytes are now always rejected, and asking for an extremely large number of results no longer exhausts memory.
- Every change is recorded in a write-ahead log before it is confirmed. After a crash or power loss, muninn recovers every confirmed change; a change that was only half written when the machine went down is dropped cleanly instead of stopping the database from opening.
- You can choose how strictly writes are protected: confirmed only once they are on disk (safest), written to disk at a short fixed interval such as every 10 ms (faster; a power cut can lose at most that interval), or left to the operating system (fastest; survives the application crashing, not the machine). When many writers save at the same time they share disk flushes, so total write throughput grows with the number of writers.
- Documents can be identified by your own ids, either numbers or text.
- Stored documents (JSON) are read directly from the data files without loading or copying them first, so opening a large collection is quick.
- Saving the list of a collection's data files is all-or-nothing: a crash during a save leaves the previous state intact, and leftover temporary or obsolete files can be cleaned up safely.
- A data directory can be opened for writing by only one program at a time; a second attempt gets a clear "already open" error instead of risking damage to the data.
- Exact vector search is much faster: it uses every CPU core and applies filters while it searches, so a filtered search costs no more than an unfiltered one.
- Fast approximate vector search (HNSW) for large collections. It builds on all CPU cores, accepts new vectors at any time (also after being reopened from disk), and on our test data finds 99% or more of the true ten nearest neighbours at its default settings.
- Filtered vector search always returns as many results as you asked for whenever that many documents match the filter, and picks the fastest way to do it automatically, including when only a handful of documents match.
- Optional vector compression (one byte or one bit per dimension) to search with less memory traffic; the best candidates are re-checked against the original vectors, so accuracy stays high.
- Vector indexes are saved as a single file and open instantly, however large they are: nothing has to be read into memory up front.
