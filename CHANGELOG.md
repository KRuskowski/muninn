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
- Index files with damaged trailing bytes are now always rejected, and asking for an extremely large number of results no longer exhausts memory.
