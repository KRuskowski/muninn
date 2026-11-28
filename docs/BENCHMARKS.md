# Benchmarks

How muninn is compared with established engines, and how to reproduce every number. The harness lives in `crates/muninn-bench`; it is a tool, never a dependency of the library, so it may use heavy dependencies (tantivy, clap, Python packages).

## One command

On a fresh Linux machine (x86-64, Rust toolchain from `rust-toolchain.toml`, `python3` with `venv`, `curl`, about 15 GB free disk and 16 GB RAM for the default run, 30 GB RAM for all 8.8M MS MARCO passages):

```bash
git clone <muninn repo> && cd muninn
crates/muninn-bench/scripts/bench_all.sh
```

This:

1. creates a Python virtualenv (`~/benchenv`) with the pinned packages from `scripts/requirements.txt` (numpy, h5py, hnswlib, faiss-cpu, usearch);
2. downloads missing datasets to `~/datasets/raw`, checks their size and SHA-256 against the values recorded in `scripts/prepare_datasets.py`, and converts them (`prepare_datasets.py`);
3. builds `muninn-bench` in release mode with `-C target-cpu=native` (`NATIVE=0` to disable);
4. checks each vector dataset's ground truth against muninn's exact search;
5. runs every engine on every dataset, writing one JSON file per run;
6. writes `REPORT.md` from the JSON files (`scripts/report.py`).

Results go to `~/bench-results/<UTC timestamp>/` (`OUT=...` to change). A failing engine is recorded in `failures.txt` and in the report, and the script exits non-zero, but the other engines still run. Every setting is an environment variable documented at the top of `bench_all.sh`; the most useful:

| variable | default | meaning |
|---|---|---|
| `CPUS` | all | CPU list every step is pinned to with `taskset`, e.g. `0-7`; the thread count follows from it |
| `DATASETS` | `sift-128-euclidean,glove-100-angular` | vector datasets; `nytimes-256-angular` is also available |
| `ENGINES` | `muninn,muninn-flat,muninn-engine,hnswlib,faiss,usearch` | vector engines |
| `M`, `EFC` | 16, 200 | HNSW graph parameters, identical for every engine |
| `EFS` | `10,16,24,32,48,64,96,128,192,256,384,512` | search `ef` sweep |
| `RUNS` | 3 | repetitions of each measurement; the best is kept |
| `MSMARCO_LIMIT` | 1000000 | passages to index; 0 for all 8,841,823 |
| `TEXT` | 1 | run the MS MARCO text benchmark |
| `LABEL` | | printed at the top of the report (e.g. "shared VM, noisy") |

Single steps can be run by hand: `target/release/muninn-bench --help` (subcommands `hnsw`, `flat`, `verify-gt`, `text`, `engine-vector`, `engine-text`, `sysinfo`) and `scripts/run_ann.py --help`.

## Datasets

| name | vectors | dim | metric | queries | source |
|---|---|---|---|---|---|
| `sift-128-euclidean` | 1,000,000 | 128 | L2 | 10,000 | TEXMEX SIFT1M, `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` (falls back to ann-benchmarks' `sift-128-euclidean.hdf5`) |
| `glove-100-angular` | 1,183,514 | 100 | cosine | 10,000 | `http://ann-benchmarks.com/glove-100-angular.hdf5` |
| `nytimes-256-angular` | 290,000 | 256 | cosine | 10,000 | `http://ann-benchmarks.com/nytimes-256-angular.hdf5` |
| `msmarco-passage` | 8,841,823 passages | | BM25 | 6,980 | `https://msmarco.z22.web.core.windows.net/msmarcoranking/` `collection.tar.gz`, `queries.tar.gz`, `qrels.dev.small.tsv` |

Vector datasets are stored as `base.fbin`, `query.fbin` (`u32` rows, `u32` dims, little-endian `f32` data) and `gt.ibin` (`u32` rows, `u32` k, `i32` neighbour ids, nearest first), with a `meta.json` that records sources, checksums and file sizes. `queries.dev.small.tsv` is not shipped in `queries.tar.gz`; it is defined as the queries of `queries.dev.tsv` whose ids appear in `qrels.dev.small.tsv` (6,980 queries), and is built that way. Passage ids are checked to equal line numbers, which the harness uses as document ids.

**Ground truth is verified, not trusted.** `muninn-bench verify-gt` recomputes the exact top-10 for 1,000 evenly spaced queries with muninn's `FlatIndex` and compares them with the provided neighbours, accepting a differing id only when it lies at the same distance as the 10th true neighbour (a tie, computed independently in `f64`). The run fails if fewer than 99% of queries agree. On 2026-11-28 all three vector datasets agreed on 100% of sampled queries counting ties; by id alone SIFT agreed on 99.2% and NYTimes on 92.0% (NYTimes has many equidistant neighbours), so id-based recall on NYTimes cannot reach 1.0 for any engine.

## Engines and fairness

Every engine runs on the same machine, pinned to the same CPUs, with the same thread count, one after another (never concurrently).

### Vector search

| engine | what is measured |
|---|---|
| `muninn-hnsw` | `HnswIndex::build` (parallel), written to a file and fsynced, then memory-mapped with `HnswIndex::open` and searched from the mapping, as a muninn segment is served |
| `muninn-hnsw-sq8` | the same graph with SQ8 codes for traversal and full-precision re-ranking (default re-rank depth); build time includes training the quantiser |
| `muninn-flat` | `FlatIndex` exact scan on the first 1,000 queries; a sanity baseline (recall 1 up to ties) |
| `muninn-engine` | the whole database, end to end: `Database::open`, a collection with an HNSW vector field (same `M`, `ef_construction`), `Collection::upsert` in batches of 1,000 (each batch durable before `upsert` returns), `flush`, `compact` to one segment, then `Collection::search` with `Query::vector(..).ef(..)` |
| `hnswlib` | `hnswlib.Index`, built from source by pip, which compiles it with `-march=native` |
| `faiss-hnsw` | `faiss.IndexHNSWFlat` from the `faiss-cpu` wheel (which dispatches to its AVX-512 or AVX2 build at runtime) |
| `usearch` | `usearch.index.Index` with `f32` storage |

- **Same graph parameters.** `M = 16` and `ef_construction = 200` for all; muninn, hnswlib and faiss all keep `2 * M` neighbours on level 0. usearch's `connectivity` is set to `M`.
- **Same sweep.** The same `ef` values; every engine searches with `max(ef, k)` candidates and returns `k = 10` results.
- **Cosine.** muninn and hnswlib normalise vectors themselves (inside the timed build and search). faiss has no cosine metric: base and query vectors are normalised with numpy before the timed regions and searched by inner product, which slightly favours faiss.
- **Build.** All threads, wall time of the call that makes the index searchable. Index size is the size of the file each engine writes (`write_to`, `save_index`, `write_index`, `save`). Resident memory growth during the build is reported where measurable (Linux `VmRSS`); it is indicative only, since allocators keep freed memory.
- **Search protocol** (identical in `src/run.rs` and `scripts/run_ann.py`), per `ef`:
  1. warm-up: every query once on all threads, untimed;
  2. one thread: every query, one at a time, on one thread; QPS = queries / wall time; best of `RUNS`;
  3. all threads: every query spread over the thread pool, one query per task; best of `RUNS`.
  For muninn, latency percentiles come from timing each query in step 2. For the Python engines step 2 is one batch call with one thread (so QPS has no per-query Python overhead), and latencies come from a separate pass of one call per query, which includes the Python-to-C++ call; the report states that overhead (the difference between the per-call mean and the batch mean).
- **Recall** is id-based recall@10 against the dataset's ground truth (the fraction of the 10 true neighbours returned), as hnswlib and faiss report it. It is slightly stricter than ann-benchmarks' distance-based recall when there are ties.

### Full-text search (BM25)

muninn and tantivy index the same passages and answer the same 6,980 MS MARCO dev queries, top 10 by BM25.

- **Same scoring.** Both use BM25 with `k1 = 1.2`, `b = 0.75`, `idf = ln(1 + (N - df + 0.5) / (df + 0.5))`, Lucene-style one-byte document-length norms, and OR semantics over distinct query terms. The report gives the overlap of the two engines' top-10 lists as a sanity check.
- **Same analysis.**

  | analyzer | muninn | tantivy |
  |---|---|---|
  | `default` | `AnalyzerConfig::default()`: alphanumeric runs, Unicode lowercase, tokens over 255 bytes dropped | `SimpleTokenizer`, `LowerCaser`, `RemoveLongFilter::limit(256)` |
  | `english` | `AnalyzerConfig::english()`: the above plus Latin diacritic folding, Lucene's English stop words, Porter stemmer | the above plus `AsciiFoldingFilter`, `StopWordFilter(English)` (the same Lucene list), `Stemmer(English)` |

  The English stemmers differ: muninn uses the original Porter algorithm, tantivy Snowball English (Porter2), which is what tantivy users get. Expect lower overlap for `english` for that reason.
- **Same index content.** tantivy indexes term frequencies without positions and stores no text, like muninn. It also keeps a `u64` fast field with the passage number, needed to map its results back; its size is reported.
- **Indexing.** muninn: one `MemTextIndex` on one thread, then written as one file and fsynced. tantivy: an `IndexWriter` with 1 GiB per indexing thread, merging disabled while indexing, one commit, then all segments merged into one and obsolete files deleted. Indexing time covers everything up to one durable segment for both (muninn: add + write; tantivy: add + commit + merge). tantivy is timed with one indexing thread (the searched index, matching muninn's single-threaded writer) and with all threads (reported as an extra line).
- **Search.** Both open their single segment memory-mapped. Each query string is analysed inside the timed region by the engine's own analyzer. tantivy runs a `BooleanQuery` of one `TermQuery` per distinct term (`BooleanQuery::new_multiterms_query`, block-WAND) collected with `TopDocs::with_limit(10).order_by_score()`; muninn runs `TextSearcher::search` (block-max MaxScore). Mapping tantivy's document addresses to passage numbers happens after timing. Same warm-up / one-thread / all-thread protocol as above.

### End to end through the engine

The component benchmarks above measure index structures in isolation. `muninn-engine` (vector) and `engine-text` (MS MARCO) measure what an application gets from the `muninn` crate:

- **Ingest** is reported in three parts: all `upsert` calls (write-ahead log, memtable, background flushes and merges; building the `Document`s is included, as an application would have to), `flush` (everything into segments) and `compact` (all segments merged into one). The default durability is `each` (every `upsert` is on stable storage before it returns); `ENGINE_DURABILITY=batch` (fsync at most 10 ms later) or `none` change that, `ENGINE_BATCH` the batch size.
- **Size** is measured after closing and reopening the database (the reopen is timed too): the files the current manifest references (vectors, graph or postings, id map, stored documents) plus the manifest and the write-ahead log. muninn keeps the previous manifest generation and its segments as a crash fallback until the next commit, so right after a full compaction the directory is temporarily about twice that; the whole directory is reported as `dir_bytes`, with a breakdown by file kind. Searches run on the reopened collection.
- **Search** uses `Collection::search` on the compacted collection with the same protocol as everything else. For text, the run also builds the component-level index on the same passages and reports the overlap of the two result lists, which must be 1.0: the engine has to rank exactly like the index it wraps.
- Hybrid search is not benchmarked yet: no standard dataset pairs embeddings with text and relevance judgements.

## Output

Each run directory contains:

- `<dataset>__<engine>__M16_efc200.json`: one vector run (build figures, the full `ef` sweep with recall, QPS on 1 and all threads, latency mean/p50/p95/p99/max);
- `<dataset>__gt-check.json`: the ground-truth verification;
- `msmarco-passage-<docs>__text-<analyzer>.json`: both text engines, indexing and search figures, and their overlap;
- `<dataset>__muninn-engine__M16_efc200.json`, `msmarco-passage-<docs>__engine-text-<analyzer>.json`: the end-to-end engine runs;
- `system.json`, `config.json`: machine (CPU model and flags, machine type, RAM, kernel), git commit, `rustc` version, `RUSTFLAGS`, CPU pinning, and every setting of the run (also embedded in each result file under `system`);
- `REPORT.md`: build tables, best QPS per recall target, the full sweep per engine with the Pareto front marked, and the text tables;
- `logs/`: the console output of every step.

## Interpreting results

- Numbers from a shared machine (other jobs running) are noisy and labelled as such; authoritative numbers come from a dedicated VM with nothing else running.
- Always read QPS next to recall. A faster point with lower recall is not a faster engine; compare at equal recall (the "best QPS at a recall target" table).
- Single-thread QPS and latency show per-query efficiency; all-thread QPS shows how well an engine scales across cores on the same machine.
