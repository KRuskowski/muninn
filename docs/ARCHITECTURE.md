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
- **Filters.** Every search takes `Option<&dyn DocFilter>`. The engine passes "user filter AND NOT deleted". Indexes must apply it *during* search, not after, so that `k` results come back whenever they exist. A filter may offer `DocFilter::candidates()`, a `BitSet` holding every passing id (possibly more); exact scans over a selective filter then visit only those ids. `BitSet` caches its population count, so `selectivity_hint()` is free after the first call.
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

### Engine implementation (crate `muninn`)

How the design above is realised; the module docs in `crates/muninn/src` hold the details.

- **Writes.** One `upsert`/`delete` call is one WAL record (hand-rolled binary, format byte 1; see `codec.rs`). Appends run concurrently and share fsyncs; records are then applied to the memtable strictly in LSN order, so memory always equals what replay produces. Live writes and recovery apply records through the same function, decoding the logged bytes.
- **Snapshots.** A `Version` lists segments (oldest first), frozen memtables and the active memtable, each with a visible length and a copy-on-write tombstone `Arc<BitSet>`. A batch publishes one new version, so no search sees part of it.
- **Memtable.** Vectors live in an append-only chunked arena that readers scan without locks (`arena.rs`, the engine's only `unsafe`). Id map, stored records and `MemTextIndex`es sit behind an `RwLock`: writers hold it while applying a batch and publishing; text queries hold the read side while taking the snapshot, gathering BM25 statistics and searching the memtable's text, so statistics always match the snapshot.
- **Ids.** No global id map. The current version of an id is found by probing sources newest first (active memtable id map, frozen memtables, segment id maps), counting a hit only if not deleted. That costs one hash lookup per source and keeps open O(segments).
- **Flush.** A full memtable is frozen with its checkpoint LSN; one worker thread per collection writes it as a segment with the same doc ids, persists changed tombstones of every segment, commits the manifest, swaps the memtable for the segment in a new version, commits the same manifest once more and then truncates the WAL below the checkpoint. The store keeps two manifest generations and falls back to the older if the newest is unreadable; the second commit makes both describe the new state, so the log and files only the state before needed can go. Persisted tombstones may include deletes logged after the checkpoint; replaying those again is harmless. While writes outpace the worker (other frozen memtables wait, or the collection already ends with graphless segments), segments of an HNSW collection are flushed with a flat index and get their graph from a merge.
- **Compaction.** Merges contiguous runs of segments, so write order across segments is kept. Policy, in order: drop fully deleted segments; give graphless segments a graph, by folding a run of them into the graph segment before it when the rewrite is proportionate (that segment at most 16 times the documents added, within `max_merge_docs`), else merging the oldest group of neighbours that fits under `max_merge_docs` and is worth it (two or more segments, or one large enough for a graph; a small segment that pairs with no neighbour is skipped and stays exact); the run at the end waits until there are `merge_factor` of them, a memtable's worth and a `merge_factor`th of that graph, or no write arrived for `merge_idle_delay`, while a run left between graph segments is handled at once; bound the segment count; size tiers of `merge_factor`; rewrite segments with many deletes. A merge whose first input has an HNSW graph and no deletes extends that graph with the other inputs' vectors instead of building a new one, so a bulk load inserts each vector into a graph about once. Every merge commits twice like a flush, so the replaced segments are deleted when it completes (on Windows, files still mapped by a running search cannot be deleted; they go at the next flush, merge or open). Deletes arriving during a merge are carried over at publish. `compact()` flushes and merges everything into one segment.
- **Search.** Per source in parallel; filter = user filter AND not deleted AND visible. Candidates merge by score, then by write order (source position, doc id), which equals the global top-k because every index breaks ties by smaller doc id. Text uses statistics summed over all sources; like Lucene they count deleted documents until compaction removes them. A multi-field text query scores a document by its best field. Hybrid runs both sides with `max(4k, 40)` candidates and fuses by RRF (default `k = 60`) or a weighted sum of min-max normalised scores.
- **Filters.** Payload filters are `muninn_filter::Filter` (re-exported as `muninn::Filter`, JSON form included), checked against the schema's filterable fields before the search. Every memtable has a `MemIndex` fed with the same doc ids as its other structures; every segment has a `pindex` file (`FrozenIndex`), merged by compaction with the same doc maps as the text indexes. Per source, the filter evaluates to a bitset (its count taken once per query for the selectivity hint), ANDed with an optional `IdFilter` (external id sets and predicates) and with "not deleted".

## Component requirements

### muninn-simd
Kernels for `dot`, `l2sq`, `distance_batch`, `dot_f32_u8` (SQ8 query x codes), `dot_i8`, `l2sq_u8`, `hamming`, chosen once at runtime: AVX-512 (F/BW/VL/VNNI where useful), AVX2+FMA, NEON, scalar. Every kernel must equal the scalar reference within float reassociation tolerance for all lengths 0..=1024 and random data. Criterion benchmarks per kernel and dimension (32, 96, 128, 384, 768, 1536).

### muninn-vector
- `FlatIndex`: blocked, multi-threaded, filter-aware exact scan; the recall oracle.
- HNSW: parallel batch build, incremental insert, filter-aware search (keep traversing through filtered-out nodes; fall back to brute force when the filter is very selective), configurable `M`, `ef_construction` and `ef`. Graph stored as flat arrays (fixed-width level-0 adjacency) so it serialises as container sections and opens zero-copy.
- Quantised codes (SQ8): ranges are fitted on clipped per-dimension quantiles of a hashed row sample and checked on a second sample; vectors far outside the ranges are marked and scored at full precision, and if the bulk itself cannot be resolved every vector is (`HnswStats::exact_rows`). Large builds search with such codes while linking (re-scored exactly before neighbour selection), or with full vectors when SQ8 cannot stand in for them.
- HNSW node order (format version 3): a bulk build into an empty index renumbers its nodes breadth-first for memory locality and keeps an internal/external id map inside the index, so callers (the engine) see exactly the `DocId`s they inserted. Nodes inserted later keep equal ids. Version 1 and 2 files still open; at open the maps are checked to be inverse permutations (one O(n) pass), else the file is rejected as corrupt. Ties: results tie-break on the caller's id over everything the beam found, but the beam itself (`ef` candidates) is ordered by internal id, so when more than `ef` vectors are exactly equidistant at the cut-off, which of them reach the results depends on the node order. Distances and scores are never `-0.0` (a zero inner product gives distance `+0.0`, a zero distance scores `+0.0`), so zeros tie like any equal values in results, `TopK` and the engine's merge.
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
