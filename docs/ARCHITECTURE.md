# muninn architecture

muninn is an embedded (in-process) vector database and full-text search engine, with an optional HTTP server. The single design goal is speed: query latency, throughput, and ingest rate, without giving up durability or recall.

This document is the contract between workstreams. If you change a contract here, say so in your PR.

## Crates

| crate | owns | depends on |
|---|---|---|
| `muninn-core` | `DocId`, `Hit`, `Metric`, `TopK`, `DocFilter`/`BitSet`, `Error`, `Region`/`AlignedBuf`, the container file format, CRC-32C | nothing |
| `muninn-simd` | distance kernels with runtime CPU dispatch | core |
| `muninn-vector` | `VectorIndex` trait, `FlatIndex`, HNSW, quantisation | core, simd |
| `muninn-text` | `TextSearcher` trait, analysis, inverted index, BM25, pruning | core |
| `muninn-store` | WAL, mmap, id map, payloads, tombstones, manifest | core |
| `muninn` | the engine: collections, segments, search, hybrid fusion | all above |
| `muninn-server` (later) | HTTP API and the `muninn` CLI | muninn |
| `muninn-bench` (later) | datasets, recall/QPS harness, comparisons | muninn |

Index crates never depend on each other or on `muninn-store`. They read from a `Region` and write through `ContainerWriter`, both in core, so the store decides where bytes live and the indexes decide what they mean.

## Shared conventions

- **Ids.** `DocId = u32`, dense from 0 within a segment, so ids index arrays and bitsets directly. User-facing ids (strings or u64) are mapped by the store.
- **Scores.** `Hit::score` is always higher-is-better. Indexes compute internal *distances* (lower is closer) and convert with `Metric::score`. See the table in `crates/muninn-core/src/metric.rs`.
- **Cosine.** Vectors are normalised once at insert; the index normalises the query. Cosine search is then one dot product per candidate.
- **Determinism.** `TopK` breaks distance ties by smaller id. Results must not depend on thread count or segment order.
- **Filters.** Every search takes `Option<&dyn DocFilter>`. The engine passes "user filter AND NOT deleted". Indexes must apply it *during* search, not after, so that `k` results come back whenever they exist.
- **Errors.** `muninn_core::Error`. Never panic on bad input or bad files; panics are for broken invariants only.
- **Files.** Every index file is a container (`muninn_core::container`): header, 64-byte-aligned sections, table and footer at the end, CRC-32C. Write it in one streaming pass; open it from a `Region` and read arrays in place with `Region::cast`. Bump your `version` on any format change. Little-endian only.
- **Unsafe.** Allowed where it buys speed (SIMD, zero-copy casts). Every `unsafe` block has a `// SAFETY:` comment, and every SIMD path is tested against the scalar reference.

## Engine design (the target the components are built for)

A **collection** has a schema (vector dimension, metric, named text fields, stored payload), a directory, a WAL, a mutable **memtable**, and a list of immutable **segments**.

```
upsert ─▶ WAL append (group commit) ─▶ memtable ─(size threshold)─▶ flush
                                                                      │
search ─▶ snapshot of {memtable, segments} ─▶ parallel per-segment     ▼
          search with filter ─▶ merge TopK ─▶ (hybrid fusion)     immutable segment
                                                                  written, fsynced,
                                                                  manifest swapped,
                                                                  WAL truncated
```

- **Memtable.** Exact `FlatIndex` vectors plus an in-memory text index. Brute force is faster than any graph below tens of thousands of vectors, and inserts are O(1).
- **Segment.** Built once from a frozen memtable or from a merge: an HNSW graph (or flat, if small), quantised codes, full-precision vectors for re-ranking, one text index per field, payloads and the id map. All files are containers, memory-mapped on open.
- **Deletes and updates.** A delete sets a bit in the segment's tombstone `BitSet`; an update is delete + insert. Tombstones are persisted next to the segment and named by the manifest.
- **Manifest.** Lists live segments and their tombstone files. Replaced atomically (write temp, fsync, rename, fsync dir). Readers hold an `Arc` snapshot, so searches never block writes and never see half a flush.
- **Compaction.** Background merge of small segments and of segments with many tombstones.
- **Search.** Fan out over segments with rayon, one `TopK` per segment, merge. Hybrid search runs the vector and text queries and fuses them with reciprocal-rank fusion (default) or a weighted, normalised score blend.

## Component requirements

### muninn-simd
Kernels for `dot`, `l2sq`, `distance_batch`, `dot_i8`, `l2sq_u8`, `hamming`, chosen once at runtime: AVX-512 (F/BW/VL/VNNI where useful), AVX2+FMA, NEON, scalar. Every kernel must equal the scalar reference within float reassociation tolerance for all lengths 0..=1024 and random data. Criterion benchmarks per kernel and dimension (32, 96, 128, 384, 768, 1536).

### muninn-vector
- `FlatIndex`: blocked, multi-threaded, filter-aware exact scan; the recall oracle.
- HNSW: parallel batch build, incremental insert, filter-aware search (keep traversing through filtered-out nodes; fall back to brute force when the filter is very selective), configurable `M`, `ef_construction` and `ef`. Graph stored as flat arrays (fixed-width level-0 adjacency) so it serialises as container sections and opens zero-copy.
- Quantisation: SQ8 (per-dimension min/max to u8 or i8) and binary (sign bits, Hamming), each with full-precision re-ranking.
- Recall tests against `FlatIndex` on clustered synthetic data.

### muninn-text
- Analyzer: Unicode-aware lowercase tokenisation, optional stop words and stemming hook. The analyzer used at index time is recorded in the file.
- Writer: in-memory postings, then a frozen container: term dictionary (sorted, with an FST or a front-coded block index), postings in blocks of 128 doc ids and frequencies, bit-packed, with per-block max scores, plus document lengths.
- Search: BM25 (k1 = 1.2, b = 0.75), OR queries, block-max WAND or MaxScore pruning, filter-aware. Correctness test: pruned results equal exhaustive BM25.

### muninn-store
- WAL: length-prefixed, CRC-checked records; group commit; configurable durability (fsync each write / each batch / never); replay that stops cleanly at a torn tail.
- `mmap` of files into `Region`s; atomic file replacement helpers.
- Id map (user id ↔ `DocId`) and payload store (JSON bytes per document), both as containers.
- Manifest with atomic replacement, and tombstone files.

## Performance discipline

- Every speed claim comes with a benchmark someone can rerun, the machine it ran on, and the commit.
- Authoritative numbers come from a dedicated GCP VM, not the shared workstation. Record machine type and CPU.
- Recall is reported next to every approximate-search throughput number. A faster index with worse recall is not faster.
