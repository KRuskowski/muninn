# muninn HTTP API

`muninn serve --data <dir>` exposes one database over HTTP/1.1 with JSON bodies. Every endpoint is under `/v1`. Responses are `application/json`.

Every request needs:

- `Host` naming an allowed host (see [Security](#security)): by default `localhost`, `127.0.0.1`, `[::1]` or the host given in `--addr`.
- `Content-Type: application/json` on every request with a body (`application/x-ndjson` is also accepted for the bulk import). Other types are refused with 415.
- `Authorization: Bearer <key>` when the server runs with an API key (`--api-key` or `MUNINN_API_KEY`), on every route except `GET /v1/health`.

## Security

The server is meant for the machine it runs on. It listens on `127.0.0.1` by default and has no users or permissions: whoever can send it requests can read and change everything.

What it defends against:

- **Web pages in a browser on the same machine.** Any web page can make the browser send requests to `127.0.0.1`. A "simple" cross-origin POST (`text/plain` or form encoding) goes out without asking permission first, and a page that switches its own domain name to 127.0.0.1 (DNS rebinding) can even read the answers. muninn refuses both: bodies must be `application/json`, which browsers only send cross-origin after a permission check that this server never grants; requests whose `Host` names a host outside the allowed list get 403, and so do requests carrying an `Origin` header of another host (including `null`).
- **Clients that hold resources.** Request headers must arrive within `--header-timeout` (10 s), which also closes idle keep-alive connections; a request body within `--body-timeout` (60 s); a bulk import may not pause longer than `--bulk-idle-timeout` (30 s). At most `--max-connections` (512) connections are open; more are closed at once. At most `--max-requests` (256) requests run at once and `--max-bulk` (2) bulk imports; beyond that the answer is 503 with `Retry-After: 1`.
- **Memory exhaustion.** A parsed request takes several times its size in memory, so bodies are limited to `--body-limit` (16 MiB), bulk lines to `--bulk-line-limit` (1 MiB), and all bodies held at once to `--max-body-in-flight` (64 MiB; each bulk import reserves about 18 MiB of it). A body announced larger than the limit is refused with 413 before it is read; a request that would exceed the in-flight budget gets 503.

What it does not do: encryption (TLS) and user accounts. To reach it from other machines, put it behind a reverse proxy that provides TLS and authentication, or at least set an API key, open the port only to trusted hosts, and add the name clients use with `--allowed-host NAME` (`--allowed-host '*'` turns the Host check off; only do that with an API key). Prefer `MUNINN_API_KEY` over `--api-key`: command lines are visible to other users of the machine.

## Errors

Every failure returns the same body, with a matching status code:

```json
{"error": {"code": "not_found", "message": "collection \"docs\" not found"}}
```

| code | status | when |
|---|---|---|
| `invalid_argument` | 400 | malformed JSON, unknown keys, a wrong vector dimension, an invalid schema or filter, an undeclared text or filter field |
| `unauthenticated` | 401 | an API key is set and the request lacks it or has a wrong one |
| `forbidden` | 403 | the `Host` or `Origin` header names a host that is not allowed |
| `not_found` | 404 | unknown collection, document or route |
| `method_not_allowed` | 405 | the route exists but not with this method |
| `request_timeout` | 408 | the body did not arrive within `--body-timeout`, or a bulk import paused longer than `--bulk-idle-timeout` |
| `already_exists` | 409 | creating a collection whose name is taken |
| `payload_too_large` | 413 | the body exceeds `--body-limit` (default 16 MiB); for the bulk endpoint, one line exceeds `--bulk-line-limit` (default 1 MiB) |
| `unsupported_media_type` | 415 | a body without `Content-Type: application/json` |
| `internal` | 500 | disk errors, corrupt data, a collection that could not be opened, a collection that stopped accepting writes after a disk failure |
| `unavailable` | 503 | too many requests, bulk imports or request bytes in flight; retry after the `Retry-After` seconds |

Messages say exactly what is wrong and where, for example `request body: unknown field \`limit\`, expected one of ... at line 1 column 21` or `filter at $.and[1].range.year: range bound must be a number, got a string`.

## Health and statistics

`GET /v1/health` returns `{"status": "ok", "version": "0.1.0"}`.

`GET /v1/stats` returns the statistics of every collection:

```json
{"collections": {"docs": {
  "live_docs": 1000, "deleted_docs": 12, "memtable_docs": 200,
  "pending_flushes": 0,
  "segments": [{"id": 3, "docs": 812, "deleted": 12, "vector_index": "hnsw"}],
  "last_lsn": 57, "durable_lsn": 57, "checkpoint_lsn": 41
}},
 "unavailable": {}}
```

- `live_docs`: documents that can be found.
- `deleted_docs`: deleted or replaced documents still taking space until compaction.
- `memtable_docs`: recent documents held in memory (already durable in the write-ahead log), not yet in an on-disk segment.
- `segments`: the immutable on-disk parts; `vector_index` is `hnsw`, `flat` or `none`.
- `last_lsn`, `durable_lsn`, `checkpoint_lsn`: sequence numbers of the last applied write, of the last write on stable storage, and of the last write contained in segments.
- `unavailable`: collections whose files could not be opened when the server started, with the reason, for example `{"old": "corrupt data: ..."}`. The other collections work normally. Every endpoint of an unavailable collection answers `500 internal` with that reason; `DELETE /v1/collections/{name}` removes it.

If a disk failure leaves a collection in an unknown state, it stops accepting writes and its write endpoints answer `500 internal` with a message saying so. Restarting the server reopens it and recovers every acknowledged write.

## Collections

### Create: `PUT /v1/collections/{name}`

Names are 1 to 64 ASCII letters, digits, `_` or `-`. The body is the schema; it cannot be changed later.

```json
{
  "vector": {"dim": 384, "metric": "cosine", "index": "hnsw", "quantization": "none"},
  "text_fields": ["title", "body"],
  "analyzer": "english",
  "filterable": {"lang": "keyword", "year": "integer", "price": "float", "meta.public": "bool"}
}
```

| key | meaning | default |
|---|---|---|
| `vector` | omit (or `null`) for a text-only collection | none |
| `vector.dim` | dimensions, 1 to 65536 (required) | |
| `vector.metric` | `cosine`, `dot` or `l2` | `cosine` |
| `vector.index` | `"hnsw"`, `"flat"`, or `{"kind": "hnsw", "m": 16, "ef_construction": 200}` | `hnsw` (m 16, ef_construction 200) |
| `vector.quantization` | `none`, `sq8` (one byte per dimension) or `binary` (one bit) | `none` |
| `text_fields` | names of the full-text fields | `[]` |
| `analyzer` | `"default"` (lowercase only), `"english"` (accent folding, stop words, Porter stemming), or an object `{"fold_diacritics": bool, "english_stop_words": bool, "stemmer": "none" \| "porter_english", "max_token_bytes": 255}` | `default` |
| `filterable` | payload fields usable in filters, as `{"field": kind}` or `[{"name": "field", "kind": kind}]`; kinds are `keyword`, `integer`, `float`, `bool`; dotted names reach nested objects | none |

A schema needs a vector field, at least one text field, or both. Responds `201 Created` with the collection description (below). `409 already_exists` if the name is taken.

### List: `GET /v1/collections`

`{"collections": ["docs", "notes"], "unavailable": {}}`: names sorted, and collections that could not be opened (see statistics above).

### Describe: `GET /v1/collections/{name}`

```json
{"name": "docs", "schema": {...every setting spelled out...}, "stats": {...as in /v1/stats...}}
```

The returned schema can be sent back to `PUT` to create an identical collection.

### Delete: `DELETE /v1/collections/{name}`

Removes the collection and its files: `{"name": "docs", "dropped": true}`.

## Documents

A document:

```json
{"id": "a1", "vector": [0.1, 0.2, 0.3], "text": {"title": "...", "body": "..."}, "payload": {"lang": "en", "year": 2024}}
```

- `id`: a string of at most 64 KiB or a non-negative integer up to 2^64 - 1. Numeric and string ids are separate: `7` and `"7"` are different documents.
- `vector`: required when the collection has a vector field, forbidden otherwise.
- `text`: any of the declared text fields; others are rejected.
- `payload`: any JSON, stored and returned as is. Filters read the declared `filterable` fields from it.

Writing a document with an existing id replaces it.

### Upsert: `POST /v1/collections/{name}/documents`

```json
{"documents": [document, ...]}
```

The batch is all-or-nothing: if any document is invalid nothing is stored, and after a crash either the whole batch is there or none of it. When the call returns, the documents are searchable and as durable as `--durability` promises. Response: `{"count": 2, "took_us": 812}`.

### Bulk import: `POST /v1/collections/{name}/documents:bulk[?batch=N]`

Newline-delimited JSON: one document per line, blank lines ignored. Send it as `Content-Type: application/x-ndjson` (or `application/json`). The body is streamed, so its total size is unlimited; one line may not exceed `--bulk-line-limit` (1 MiB), and the client may not pause longer than `--bulk-idle-timeout`. At most `--max-bulk` imports run at once. Documents are stored in batches of `N` (default `--bulk-batch`, 1000; at most 100000), each all-or-nothing; reading, parsing (on all cores) and writing overlap. Response: `{"count": 50000, "batches": 50, "took_us": 1234567}`.

If a line is invalid, the request fails with an error naming the line, and the batches before it stay stored, as the message says: `line 1203: vector: expected a number, got "x" at line 1 column 30; the 1000 documents before it were stored` (or `nothing was stored`).

```sh
curl -sS -X POST -H 'Content-Type: application/x-ndjson' --data-binary @docs.jsonl \
  'http://127.0.0.1:7700/v1/collections/docs/documents:bulk'
```

### Delete: `DELETE /v1/collections/{name}/documents`

`{"ids": ["a1", 7]}`. Unknown ids are ignored; all-or-nothing like an upsert. Response: `{"count": 2, "took_us": 95}` (the number of ids received).

### Get: `GET /v1/collections/{name}/documents/{id}[?id_type=string|u64]`

Returns the document (`vector` is normalised for cosine collections; `payload` is left out when the document has none). A path segment made only of digits is a numeric id; add `?id_type=string` to fetch a string id such as `"12"`. `404 not_found` if there is no such document.

## Search: `POST /v1/collections/{name}/search`

```json
{
  "vector": [0.1, 0.2, 0.3],
  "text": "fast embedded database",
  "k": 10,
  "filter": {"and": [{"eq": {"lang": "en"}}, {"range": {"year": {"gte": 2020}}}]},
  "fusion": {"rrf": {"k": 60}},
  "with_payload": true,
  "fields": ["title"]
}
```

`vector` only is a nearest-neighbour search, `text` only a keyword (BM25) search, both a hybrid search.

| key | meaning | default |
|---|---|---|
| `vector` | query vector, same dimension as the collection | |
| `text` | query words; a document matches if it contains any of them | |
| `k` | number of results, 1 to 10000 | 10 |
| `filter` | payload filter (below) | none |
| `ids` | only these document ids, e.g. `[1, "a2"]` | all |
| `fusion` | hybrid only: `"rrf"`, `{"rrf": {"k": 60}}` (reciprocal-rank fusion) or `{"weighted": {"vector": 0.7, "text": 0.3}}` (weighted sum of min-max normalised scores) | `{"rrf": {"k": 60}}` |
| `with_payload` | include each hit's payload | `false` |
| `fields` | text fields to search (at least one); a document scores its best field | all |
| `ef` | HNSW candidate-list size; larger is more accurate and slower | 64 |
| `rerank` | quantised candidates re-scored exactly | 4k (sq8), 16k (binary) |
| `candidates` | hybrid only: candidates per side before fusion | max(4k, 40) |

Response:

```json
{"hits": [
  {"id": "a1", "score": 0.0325, "vector_score": 0.91, "text_score": 7.2, "payload": {"lang": "en"}}
], "took_us": 412}
```

- `score`: higher is better. The vector similarity for vector searches (cosine: the cosine similarity; dot: the dot product; l2: minus the squared Euclidean distance), BM25 for text searches, the fused score for hybrid searches.
- `vector_score`, `text_score`: each side's score when that side found the document, else `null`.
- `payload`: only with `with_payload: true`.
- `took_us`: server time to parse the request and search, in microseconds.

Equal scores come back in the order the documents were written. Filters are applied during the search, so `k` hits come back whenever `k` documents match.

### Filters

```text
filter := {"eq":     {FIELD: VALUE}}
        | {"in":     {FIELD: [VALUE, ...]}}
        | {"range":  {FIELD: {"gt" | "gte" | "lt" | "lte": NUMBER, ...}}}
        | {"exists": FIELD}
        | {"and":    [filter, ...]}
        | {"or":     [filter, ...]}
        | {"not":    filter}
```

Fields must be declared `filterable` in the schema. A field holding a list matches when any element matches. A payload value of the wrong kind (text where a number is declared) counts as missing. `and []` matches everything, `or []` and `in []` match nothing.

## Maintenance

`POST /v1/collections/{name}/flush` writes the documents held in memory to an on-disk segment and shortens the write-ahead log. Not needed for durability; useful before a backup or to make the next start faster. Response: `{"took_us": 1520}`.

`POST /v1/collections/{name}/compact` flushes, then merges all segments into one, removing deleted and replaced documents for good. Searches and writes continue meanwhile. Response: `{"took_us": 250000}`.

## Shutdown

On Ctrl-C (SIGINT) or SIGTERM the server stops accepting connections, lets the requests in progress finish for up to `--shutdown-timeout` (30 s) and closes the connections still open after that. It then writes every collection's in-memory documents to disk and closes the database. A second signal stops waiting for open requests at once (the data is still written to disk); a third exits immediately, which is safe because every acknowledged write is already in the write-ahead log.

## Example

```sh
muninn serve --data ./data &
J='Content-Type: application/json'
curl -sS -X PUT -H "$J" localhost:7700/v1/collections/docs \
  -d '{"vector": {"dim": 3}, "text_fields": ["title"], "filterable": {"lang": "keyword"}}'
curl -sS -X POST -H "$J" localhost:7700/v1/collections/docs/documents \
  -d '{"documents": [{"id": 1, "vector": [0.1, 0.2, 0.3], "text": {"title": "hello world"}, "payload": {"lang": "en"}}]}'
curl -sS -X POST -H "$J" localhost:7700/v1/collections/docs/search \
  -d '{"vector": [0.1, 0.2, 0.3], "text": "hello", "filter": {"eq": {"lang": "en"}}, "with_payload": true}'
```
