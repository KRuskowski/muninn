#!/usr/bin/env python3
"""Runs a competitor ANN engine with the harness's protocol and schema.

Engines: hnswlib (hnswlib.Index), faiss (IndexHNSWFlat), usearch (Index,
f32). Every engine gets the same M, ef_construction, ef sweep, k and
thread count as muninn, and writes the JSON schema `muninn-bench hnsw`
writes (see src/vector.rs).

Measurement, per ef (mirrors src/run.rs):
  1. warm-up: all queries in one batch call on all threads, untimed;
  2. one thread: all queries in one batch call with one thread, best of
     --runs; this gives qps_1t with no per-query Python overhead;
  3. latency: one call per query on one thread, timed around the call
     (so it includes the Python-to-C++ call, reported separately as
     `python_call_overhead_us`, the per-query difference to step 2);
  4. all threads: one batch call on all threads, best of --runs.
Recall is computed from the step 2 results.
"""

import argparse
import importlib.metadata
import json
import os
import socket
import sys
import time

import numpy as np

FLAGS = ["sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw",
         "avx512vl", "avx512dq", "avx512_vnni", "avx512_bf16", "avx512_fp16",
         "avx_vnni", "amx_tile"]
DEFAULT_EFS = "10,16,24,32,48,64,96,128,192,256,384,512"

def log(msg):
  print(msg, file=sys.stderr, flush=True)

def read_bin(path, dtype):
  with open(path, "rb") as f:
    n, d = np.fromfile(f, dtype="<u4", count=2)
    a = np.fromfile(f, dtype=dtype, count=int(n) * int(d))
  return a.reshape(int(n), int(d))

def status_kib(field):
  try:
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith(field):
          return int(line.split()[1]) * 1024
  except OSError:
    pass
  return None

def system_info():
  """Same fields as SystemInfo in src/sys.rs."""
  def read(p):
    try:
      with open(p) as f:
        return f.read()
    except OSError:
      return ""
  cpuinfo = read("/proc/cpuinfo").splitlines()
  def field(name):
    for line in cpuinfo:
      if line.startswith(name) and ":" in line:
        return line.split(":", 1)[1].strip()
    return None
  present = set((field("flags") or "").split())
  mem = None
  for line in read("/proc/meminfo").splitlines():
    if line.startswith("MemTotal:"):
      mem = int(line.split()[1]) * 1024
  env = lambda k: os.environ.get(k) or None
  return {
    "hostname": socket.gethostname(),
    "cpu_model": field("model name") or "unknown",
    "cpu_flags": [f for f in FLAGS if f in present],
    "available_cpus": len(os.sched_getaffinity(0)),
    "cpu_pinning": env("BENCH_CPUS"),
    "mem_total_bytes": mem,
    "kernel": read("/proc/sys/kernel/osrelease").strip(),
    "machine_type": env("BENCH_MACHINE_TYPE"),
    "git_commit": env("BENCH_GIT_COMMIT"),
    "rustflags": env("BENCH_RUSTFLAGS"),
    "rustc": env("BENCH_RUSTC"),
    "label": env("BENCH_LABEL"),
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "timestamp": int(time.time()),
  }

def latency_summary(ns):
  """Nearest-rank percentiles in microseconds, as src/measure.rs."""
  s = np.sort(np.asarray(ns, dtype=np.int64))
  def pct(p):
    rank = int(np.ceil(p / 100.0 * len(s)))
    return float(s[min(max(rank, 1), len(s)) - 1]) / 1e3
  return {"mean_us": float(s.mean()) / 1e3, "p50_us": pct(50),
          "p95_us": pct(95), "p99_us": pct(99), "max_us": float(s[-1]) / 1e3}

def recall_at_k(ids, gt, k):
  found = 0
  for got, truth in zip(ids, gt[:, :k]):
    found += len(set(int(x) for x in got[:k] if x >= 0) &
                 set(int(x) for x in truth))
  return found / (len(ids) * k)

def version(pkg):
  try:
    return importlib.metadata.version(pkg)
  except importlib.metadata.PackageNotFoundError:
    return "unknown"

class Hnswlib:
  name = "hnswlib"

  def __init__(self, a, metric, dim):
    import hnswlib
    self.lib = hnswlib
    self.space = {"l2": "l2", "cosine": "cosine", "dot": "ip"}[metric]
    self.a, self.dim = a, dim
    self.version = version("hnswlib")
    self.params = {"M": a.m, "ef_construction": a.efc,
                   "level0_neighbours": 2 * a.m, "space": self.space,
                   "random_seed": 100}
    self.notes = ["hnswlib built from source by pip (compiled with "
                  "-march=native by its setup.py)."]

  def build(self, base, threads):
    self.idx = self.lib.Index(space=self.space, dim=self.dim)
    self.idx.init_index(max_elements=len(base), ef_construction=self.a.efc,
                        M=self.a.m, random_seed=100)
    self.idx.add_items(base, np.arange(len(base)), num_threads=threads)

  def save(self, path):
    self.idx.save_index(path)

  def set_ef(self, ef):
    self.idx.set_ef(ef)

  def search(self, q, k, threads):
    labels, _ = self.idx.knn_query(q, k=k, num_threads=threads)
    return labels

class Faiss:
  name = "faiss-hnsw"

  def __init__(self, a, metric, dim):
    import faiss
    self.faiss = faiss
    self.a, self.dim = a, dim
    self.normalize = metric == "cosine"
    self.metric = (faiss.METRIC_L2 if metric == "l2"
                   else faiss.METRIC_INNER_PRODUCT)
    self.version = faiss.__version__
    self.params = {"M": a.m, "ef_construction": a.efc,
                   "level0_neighbours": 2 * a.m,
                   "metric": "L2" if metric == "l2" else "inner_product",
                   "compile_options": faiss.get_compile_options()}
    self.notes = ["faiss IndexHNSWFlat from the faiss-cpu wheel."]
    if self.normalize:
      self.notes.append("Cosine: base and query vectors are L2-normalised "
                        "with numpy before (outside) the timed regions.")

  def prepare(self, x):
    if self.normalize:
      x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    return np.ascontiguousarray(x, dtype=np.float32)

  def build(self, base, threads):
    self.faiss.omp_set_num_threads(threads)
    self.idx = self.faiss.IndexHNSWFlat(self.dim, self.a.m, self.metric)
    self.idx.hnsw.efConstruction = self.a.efc
    self.idx.add(base)

  def save(self, path):
    self.faiss.write_index(self.idx, path)

  def set_ef(self, ef):
    self.idx.hnsw.efSearch = ef

  def search(self, q, k, threads):
    self.faiss.omp_set_num_threads(threads)
    _, labels = self.idx.search(q, k)
    return labels

class Usearch:
  name = "usearch"

  def __init__(self, a, metric, dim):
    import usearch.index
    self.mod = usearch.index
    self.a, self.dim = a, dim
    # Like faiss: cosine data is normalised up front and searched by inner
    # product. usearch's "cos" metric recomputes both norms per distance
    # and was 2-3x slower in our tests, which says nothing about its graph.
    self.normalize = metric == "cosine"
    self.metric = {"l2": "l2sq", "cosine": "ip", "dot": "ip"}[metric]
    self.version = version("usearch")
    probe = self.mod.Index(ndim=dim, metric=self.metric, dtype="f32")
    self.params = {"connectivity": a.m, "expansion_add": a.efc,
                   "metric": self.metric, "dtype": "f32",
                   "hardware_acceleration": probe.hardware_acceleration}
    self.notes = ["usearch Index with f32 storage (no quantisation)."]
    if self.normalize:
      self.notes.append("Cosine: base and query vectors are L2-normalised "
                        "with numpy before (outside) the timed regions and "
                        "searched by inner product.")

  def prepare(self, x):
    return Faiss.prepare(self, x)

  def build(self, base, threads):
    self.idx = self.mod.Index(ndim=self.dim, metric=self.metric, dtype="f32",
                              connectivity=self.a.m,
                              expansion_add=self.a.efc)
    self.idx.add(np.arange(len(base)), base, threads=threads)

  def save(self, path):
    self.idx.save(path)

  def set_ef(self, ef):
    self.idx.expansion_search = ef

  def search(self, q, k, threads):
    m = self.idx.search(q, k, threads=threads)
    keys = np.asarray(m.keys)
    return keys.reshape(len(q), -1) if keys.ndim == 1 else keys

ENGINES = {"hnswlib": Hnswlib, "faiss": Faiss, "usearch": Usearch}

def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
  ap.add_argument("--data", required=True)
  ap.add_argument("--m", type=int, default=16)
  ap.add_argument("--efc", type=int, default=200)
  ap.add_argument("--ef", default=DEFAULT_EFS)
  ap.add_argument("--k", type=int, default=10)
  ap.add_argument("--threads", type=int, default=0)
  ap.add_argument("--runs", type=int, default=1)
  ap.add_argument("--queries", type=int, default=None)
  ap.add_argument("--work-dir", default="bench-work")
  ap.add_argument("--out-dir", default="bench-results")
  a = ap.parse_args()
  threads = a.threads or len(os.sched_getaffinity(0))
  with open(os.path.join(a.data, "meta.json")) as f:
    meta = json.load(f)
  base = read_bin(os.path.join(a.data, "base.fbin"), "<f4")
  query = read_bin(os.path.join(a.data, "query.fbin"), "<f4")
  gt = read_bin(os.path.join(a.data, "gt.ibin"), "<i4")
  if a.queries:
    query, gt = query[:a.queries], gt[:a.queries]
  metric = meta["metric"]
  eng = ENGINES[a.engine](a, metric, base.shape[1])
  if hasattr(eng, "prepare"):
    base, query = eng.prepare(base), eng.prepare(query)
  name = meta["name"]
  log(f"{name}: {base.shape[0]} x {base.shape[1]} ({metric}), "
      f"{len(query)} queries; {eng.name} {eng.version} M={a.m} "
      f"efC={a.efc} on {threads} threads")

  rss0 = status_kib("VmRSS:")
  t = time.perf_counter()
  eng.build(base, threads)
  build_s = time.perf_counter() - t
  rss1 = status_kib("VmRSS:")
  os.makedirs(a.work_dir, exist_ok=True)
  path = os.path.join(a.work_dir,
                      f"{name}__{eng.name}__M{a.m}_efc{a.efc}.idx")
  t = time.perf_counter()
  eng.save(path)
  with open(path, "rb+") as f:
    os.fsync(f.fileno())
  write_s = time.perf_counter() - t
  index_bytes = os.path.getsize(path)
  os.remove(path)
  log(f"  built in {build_s:.1f} s, file {index_bytes / 1e6:.1f} MB")

  k, nq, results = a.k, len(query), []
  for ef in [int(x) for x in a.ef.split(",") if x.strip()]:
    eng.set_ef(ef)
    eng.search(query, k, threads)
    qps_1t, ids = 0.0, None
    for _ in range(max(a.runs, 1)):
      t = time.perf_counter()
      got = eng.search(query, k, 1)
      qps = nq / (time.perf_counter() - t)
      if qps > qps_1t:
        qps_1t, ids = qps, got
    lat = np.empty(nq, dtype=np.int64)
    for i in range(nq):
      q = query[i:i + 1]
      s = time.perf_counter_ns()
      eng.search(q, k, 1)
      lat[i] = time.perf_counter_ns() - s
    qps_nt, same = 0.0, True
    for _ in range(max(a.runs, 1)):
      t = time.perf_counter()
      got = eng.search(query, k, threads)
      qps_nt = max(qps_nt, nq / (time.perf_counter() - t))
      same &= bool(np.array_equal(got, ids))
    latency = latency_summary(lat)
    recall = recall_at_k(ids, gt, k)
    overhead = latency["mean_us"] - 1e6 / qps_1t
    log(f"  ef {ef:4}: recall {recall:.4f}  1t {qps_1t:9.0f} q/s  "
        f"{threads}t {qps_nt:9.0f} q/s  p50 {latency['p50_us']:7.1f} us  "
        f"p99 {latency['p99_us']:7.1f} us")
    results.append({"ef": ef, "recall": recall, "qps_1t": qps_1t,
                    "qps_nt": qps_nt, "latency": latency,
                    "deterministic": same,
                    "python_call_overhead_us": overhead})

  report = {
    "kind": "vector",
    "engine": eng.name,
    "engine_version": eng.version,
    "dataset": {"name": name, "metric": metric, "n": int(base.shape[0]),
                "dim": int(base.shape[1]), "queries": nq,
                "meta": {x: meta[x] for x in meta
                         if x not in ("name", "metric")}},
    "k": k,
    "threads": threads,
    "params": eng.params,
    "build": {
      "seconds": build_s, "threads": threads, "index_bytes": index_bytes,
      "rss_delta_bytes": (rss1 - rss0) if rss0 and rss1 else None,
      "peak_rss_bytes": status_kib("VmHWM:"),
      "details": {"write_seconds": write_s},
    },
    "results": results,
    "system": system_info(),
    "notes": eng.notes + [
      "qps_1t: one batch call with one thread; latency: one call per "
      "query, including the Python call (see python_call_overhead_us)."],
  }
  os.makedirs(a.out_dir, exist_ok=True)
  out = os.path.join(a.out_dir, f"{name}__{eng.name}__M{a.m}_efc{a.efc}.json")
  with open(out, "w") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
  print(out)

if __name__ == "__main__":
  main()
