#!/usr/bin/env bash
# Runs the whole muninn benchmark comparison with one command: prepares
# the datasets (downloading what is missing), builds the harness, runs
# every engine on every dataset and writes JSON results plus REPORT.md.
#
#   crates/muninn-bench/scripts/bench_all.sh
#
# Settings come from environment variables (defaults in brackets):
#   DATA       dataset root                          [~/datasets]
#   OUT        result directory               [~/bench-results/<UTC time>]
#   WORK       scratch space for index files        [~/bench-work]
#   VENV       Python virtualenv, created if missing [~/benchenv]
#   DATASETS   vector datasets
#              [sift-128-euclidean,glove-100-angular]; also
#              nytimes-256-angular
#   ENGINES    vector engines [muninn,muninn-flat,hnswlib,faiss,usearch]
#   QUANT      muninn traversal variants on the same graph [none,sq8]
#   M, EFC     HNSW M and ef_construction                  [16, 200]
#   EFS        search ef sweep  [10,16,24,32,48,64,96,128,192,256,384,512]
#   K          results per query                           [10]
#   RUNS       repetitions per measurement, best kept      [3]
#   QUERIES    limit vector queries (empty: all)           []
#   FLAT_QUERIES  queries for the exact-scan baseline      [1000]
#   TEXT       run the MS MARCO text benchmark (0/1)       [1]
#   MSMARCO_LIMIT  passages to index, 0 for all 8.8M       [1000000]
#   ANALYZERS  text analyzers                    [default english]
#   TANTIVY_INDEX_THREADS  tantivy indexing threads; the first
#              builds the searched index           [1,<threads>]
#   CPUS       CPU list to pin everything to, e.g. 0-7 (empty: all) []
#   NATIVE     build with -C target-cpu=native (0/1)       [1]
#   LABEL      label printed at the top of the report      []
#   GIT_COMMIT commit to record when there is no .git      []
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
DATA=${DATA:-$HOME/datasets}
OUT=${OUT:-$HOME/bench-results/$(date -u +%Y%m%dT%H%M%SZ)}
WORK=${WORK:-$HOME/bench-work}
VENV=${VENV:-$HOME/benchenv}
DATASETS=${DATASETS:-sift-128-euclidean,glove-100-angular}
ENGINES=${ENGINES:-muninn,muninn-flat,hnswlib,faiss,usearch}
QUANT=${QUANT:-none,sq8}
M=${M:-16}
EFC=${EFC:-200}
EFS=${EFS:-10,16,24,32,48,64,96,128,192,256,384,512}
K=${K:-10}
RUNS=${RUNS:-3}
QUERIES=${QUERIES:-}
FLAT_QUERIES=${FLAT_QUERIES:-1000}
TEXT=${TEXT:-1}
MSMARCO_LIMIT=${MSMARCO_LIMIT:-1000000}
ANALYZERS=${ANALYZERS:-default english}
CPUS=${CPUS:-}
NATIVE=${NATIVE:-1}
LABEL=${LABEL:-}

if [ -n "$CPUS" ]; then
  pin=(taskset -c "$CPUS")
else
  pin=()
fi
threads=$("${pin[@]}" nproc)
TANTIVY_INDEX_THREADS=${TANTIVY_INDEX_THREADS:-1,$threads}

mkdir -p "$OUT/logs" "$WORK"
: > "$OUT/failures.txt"
say() { printf '\n=== %s\n' "$*" >&2; }

# Runs a named step, logging to $OUT/logs/<name>.log. A failing step is
# recorded and the run goes on, so one broken engine does not hide the
# others; the script exits non-zero at the end.
step() {
  local name=$1; shift
  say "$name"
  if ! "$@" 2>&1 | tee "$OUT/logs/$name.log" >&2; then
    echo "$name" >> "$OUT/failures.txt"
    echo "FAILED: $name (see logs/$name.log)" >&2
  fi
}

# Python environment.
if [ ! -x "$VENV/bin/python" ]; then
  say "creating $VENV"
  python3 -m venv "$VENV" || exit 1
fi
"$VENV/bin/pip" install -q -r "$here/requirements.txt" || exit 1
py=("$VENV/bin/python" -I)

# Datasets.
want=$DATASETS
[ "$TEXT" = 1 ] && want="$want,msmarco-passage"
say "preparing datasets in $DATA"
"${py[@]}" "$here/prepare_datasets.py" --root "$DATA" --datasets "$want" \
  || exit 1

# Build.
flags=${RUSTFLAGS:-}
[ "$NATIVE" = 1 ] && flags="$flags -C target-cpu=native"
export RUSTFLAGS="${flags# }"
say "building muninn-bench (RUSTFLAGS='$RUSTFLAGS')"
(cd "$repo" && "${pin[@]}" cargo build --release --locked -p muninn-bench) \
  || exit 1
bin=$repo/target/release/muninn-bench

# What the result files record about this run.
commit=${GIT_COMMIT:-$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)}
export BENCH_GIT_COMMIT=${commit:-unknown}
export BENCH_RUSTFLAGS=$RUSTFLAGS
export BENCH_RUSTC=$(cd "$repo" && rustc --version)
export BENCH_CPUS=$CPUS
export BENCH_LABEL=$LABEL
export BENCH_MACHINE_TYPE=$(curl -s -m 2 -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/machine-type \
  2>/dev/null | sed 's|.*/||')
"$bin" sysinfo > "$OUT/system.json"
cat > "$OUT/config.json" <<JSON
{"datasets": "$DATASETS", "engines": "$ENGINES", "quant": "$QUANT",
 "M": $M, "ef_construction": $EFC, "efs": "$EFS", "k": $K,
 "runs": $RUNS, "queries": "$QUERIES", "flat_queries": $FLAT_QUERIES,
 "threads": $threads, "cpus": "$CPUS", "text": $TEXT,
 "msmarco_limit": $MSMARCO_LIMIT, "analyzers": "$ANALYZERS",
 "tantivy_index_threads": "$TANTIVY_INDEX_THREADS",
 "rustflags": "$RUSTFLAGS", "git_commit": "$BENCH_GIT_COMMIT"}
JSON

common=(--threads "$threads" --runs "$RUNS" --k "$K" --work-dir "$WORK"
  --out-dir "$OUT")
qlimit=()
[ -n "$QUERIES" ] && qlimit=(--queries "$QUERIES")

IFS=, read -ra dsets <<< "$DATASETS"
IFS=, read -ra engines <<< "$ENGINES"
for ds in "${dsets[@]}"; do
  dir=$DATA/$ds
  step "$ds-gt-check" "${pin[@]}" "$bin" verify-gt --data "$dir" \
    "${common[@]}"
  for e in "${engines[@]}"; do
    case $e in
      muninn)
        step "$ds-muninn" "${pin[@]}" "$bin" hnsw --data "$dir" --m "$M" \
          --efc "$EFC" --ef "$EFS" --quant "$QUANT" "${qlimit[@]}" \
          "${common[@]}" ;;
      muninn-flat)
        step "$ds-muninn-flat" "${pin[@]}" "$bin" flat --data "$dir" \
          --queries "$FLAT_QUERIES" "${common[@]}" ;;
      hnswlib|faiss|usearch)
        step "$ds-$e" "${pin[@]}" "${py[@]}" "$here/run_ann.py" \
          --engine "$e" --data "$dir" --m "$M" --efc "$EFC" --ef "$EFS" \
          "${qlimit[@]}" "${common[@]}" ;;
      *) echo "unknown engine $e" >&2; echo "$ds-$e" >> "$OUT/failures.txt" ;;
    esac
  done
done

if [ "$TEXT" = 1 ]; then
  limit=()
  [ "$MSMARCO_LIMIT" != 0 ] && limit=(--limit "$MSMARCO_LIMIT")
  for a in $ANALYZERS; do
    step "msmarco-$a" "${pin[@]}" "$bin" text \
      --collection "$DATA/msmarco-passage/collection.tsv" \
      --queries "$DATA/msmarco-passage/queries.dev.small.tsv" \
      "${limit[@]}" --analyzer "$a" \
      --tantivy-index-threads "$TANTIVY_INDEX_THREADS" "${common[@]}"
  done
fi

say "writing $OUT/REPORT.md"
"${py[@]}" "$here/report.py" "$OUT" > "$OUT/REPORT.md"
if [ -s "$OUT/failures.txt" ]; then
  { echo; echo "## Failed steps"; echo
    sed 's/^/- /' "$OUT/failures.txt"; } >> "$OUT/REPORT.md"
  echo "some steps failed:" >&2
  cat "$OUT/failures.txt" >&2
  exit 1
fi
say "done: $OUT"
