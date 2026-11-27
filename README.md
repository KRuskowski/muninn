# muninn

A local, embedded vector database and full-text search engine written in Rust, built for speed.

muninn stores documents that carry a vector, text fields and a JSON payload, and finds them by vector similarity, by keyword relevance (BM25), or by both at once (hybrid search). It runs inside your process, or as a small HTTP server.

**Status:** early development. The building blocks are being written; see `docs/ARCHITECTURE.md` for the design and `CHANGELOG.md` for what works today.

## Building

```sh
cargo build --release
scripts/check.sh   # format, lints, tests
```

## License

MIT, see `LICENSE`.
