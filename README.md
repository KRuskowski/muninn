# muninn

A local, embedded vector database and full-text search engine written in Rust, built for speed.

muninn stores documents that carry a vector, text fields and a JSON payload, and finds them by vector similarity, by keyword relevance (BM25), or by both at once (hybrid search). It runs inside your process, or as a small HTTP server.

**Status:** early development, not yet released. Every part works and is tested on Linux and Windows; see `docs/ARCHITECTURE.md` for the design and `CHANGELOG.md` for what works today.

## Building

```sh
cargo build --release
scripts/check.sh   # format, lints, tests
```

## Usage

The `muninn` command runs the HTTP server or works on a data directory directly.

```sh
# HTTP/JSON server on 127.0.0.1 (see docs/API.md for every endpoint and
# the security model; set MUNINN_API_KEY before exposing it further)
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

## Performance

Measured on a dedicated GCP c4-standard-16 (16 vCPUs, Xeon Platinum 8581C) on 2026-11-28, commit ac3828c, against hnswlib 0.8.0, faiss 1.16.0, usearch 2.26.6 and tantivy 0.27.0, all on the same machine with the same settings (HNSW M=16, ef_construction=200, k=10). Full report, raw JSON and the exact protocol: `crates/muninn-bench/results/2026-11-28-dedicated-vm/REPORT.md` and `docs/BENCHMARKS.md`. Rerun with `crates/muninn-bench/scripts/bench_all.sh`.

Vector search, queries per second at a recall@10 target (one thread / 16 threads):

| dataset, recall | muninn | muninn SQ8 | faiss | hnswlib | usearch |
|---|---|---|---|---|---|
| SIFT1M, ≥ 0.95 | 15,806 / 164,146 | 22,465 / 246,086 | 10,172 / 89,752 | 8,644 / 77,839 | 5,401 / 53,381 |
| SIFT1M, ≥ 0.99 | 6,920 / 71,800 | 10,036 / 110,341 | 5,199 / 49,040 | 3,343 / 30,493 | 2,128 / 20,775 |
| GloVe-100, ≥ 0.9 | 1,944 / 20,797 | 2,608 / 29,151 | 1,676 / 17,616 | 1,718 / 16,260 | 1,098 / 11,743 |
| NYTimes-256, ≥ 0.9 | 3,176 / 31,933 | 5,476 / 57,882 | 1,968 / 20,436 | 2,146 / 19,408 | 1,168 / 11,976 |

Keyword search (BM25 top 10) on 1M MS MARCO passages, queries per second (one thread / 16 threads) and 99th-percentile latency:

| analyzer | muninn | tantivy |
|---|---|---|
| default | 1,990 / 19,695, p99 2.3 ms | 1,089 / 12,023, p99 5.5 ms |
| English (stop words, stemming) | 2,444 / 24,019, p99 1.9 ms | 1,576 / 17,349, p99 4.2 ms |

Through the whole database (durable inserts, flush, compaction, then search), vector search runs 2-7% and keyword search about 2% slower than the bare indexes above. Index build with 16 threads: SIFT1M 21.9 s (faiss 28.0 s, hnswlib 27.5 s); on GloVe and NYTimes muninn is currently slower to build than faiss and hnswlib (42.2 s vs 35.1 s and 38.7 s; 16.5 s vs 12.8 s and 11.9 s).

## License

MIT, see `LICENSE`.
