# muninn

A local, embedded vector database and full-text search engine written in Rust, built for speed.

muninn stores documents that carry a vector, text fields and a JSON payload, and finds them by vector similarity, by keyword relevance (BM25), or by both at once (hybrid search). It runs inside your process, or as a small HTTP server.

**Status:** early development. The building blocks are being written; see `docs/ARCHITECTURE.md` for the design and `CHANGELOG.md` for what works today.

## Building

```sh
cargo build --release
scripts/check.sh   # format, lints, tests
```

## Usage

The `muninn` command runs the HTTP server or works on a data directory directly.

```sh
# HTTP/JSON server (see docs/API.md for every endpoint)
muninn serve --data ./data --addr 127.0.0.1:7700 [--durability each|batch|none]

# Import newline-delimited JSON, creating the collection from a schema file
muninn import --data ./data --collection docs --create-schema schema.json docs.jsonl

# Search: text, vector (from a JSON file), or both, with an optional filter
muninn search --data ./data --collection docs --text "fast database" \
  --vector-file query.json --filter '{"eq":{"lang":"en"}}' -k 10

# Statistics of every collection, and merging a collection's on-disk parts
muninn stats --data ./data
muninn compact --data ./data --collection docs
```

A schema file, and one line of a JSONL file:

```json
{"vector": {"dim": 3, "metric": "cosine"}, "text_fields": ["title"], "filterable": {"lang": "keyword"}}
```

```json
{"id": 1, "vector": [0.1, 0.2, 0.3], "text": {"title": "hello world"}, "payload": {"lang": "en"}}
```

`--durability each` (the default) confirms a write once it is on disk; `batch` writes to disk at least every `--sync-interval-ms` (10 ms); `none` leaves it to the operating system. `muninn <command> --help` lists every option. The direct commands (`import`, `search`, `stats`, `compact`) need the data directory to themselves, so stop the server first or use the HTTP API.

## License

MIT, see `LICENSE`.
