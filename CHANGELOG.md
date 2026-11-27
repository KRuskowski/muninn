# Changelog

All notable changes to muninn are recorded here, written for people who use muninn rather than people who read its source.

## Unreleased

- Project started, under the MIT license.
- The foundation is in place: a shared file format with corruption detection, exact (brute-force) vector search with cosine, dot-product and Euclidean similarity, and filtering of results by document.
- Every change is recorded in a write-ahead log before it is confirmed. After a crash or power loss, muninn recovers every confirmed change; a change that was only half written when the machine went down is dropped cleanly instead of stopping the database from opening.
- You can choose how strictly writes are protected: confirmed only once they are on disk (safest), written to disk at a short fixed interval such as every 10 ms (faster; a power cut can lose at most that interval), or left to the operating system (fastest; survives the application crashing, not the machine). When many writers save at the same time they share disk flushes, so total write throughput grows with the number of writers.
- Documents can be identified by your own ids, either numbers or text.
- Stored documents (JSON) are read directly from the data files without loading or copying them first, so opening a large collection is quick.
- Saving the list of a collection's data files is all-or-nothing: a crash during a save leaves the previous state intact, and leftover temporary or obsolete files can be cleaned up safely.
- A data directory can be opened for writing by only one program at a time; a second attempt gets a clear "already open" error instead of risking damage to the data.
