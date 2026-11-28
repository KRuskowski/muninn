#!/usr/bin/env python3
"""Downloads, verifies and converts the benchmark datasets.

Layout under --root (default ~/datasets):

  raw/<source>/...            downloads, checked against known SHA-256
  <name>/base.fbin            vectors to index   (u32 n, u32 d, f32 data)
  <name>/query.fbin           query vectors
  <name>/gt.ibin              exact neighbours   (u32 n, u32 k, i32 ids)
  <name>/meta.json            metric, sources, checksums, sizes
  msmarco-passage/collection.tsv, queries.dev.small.tsv, meta.json

Vector datasets: sift-128-euclidean (TEXMEX SIFT1M, with the
ann-benchmarks HDF5 as fallback), glove-100-angular, nytimes-256-angular.
Text: MS MARCO passage ranking collection; queries.dev.small.tsv is
queries.dev.tsv restricted to the 6980 query ids of qrels.dev.small.tsv,
which is how the official small dev set is defined.

Run with `python3 -I` so nothing from the download directory is imported.
Already prepared datasets are skipped.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

import numpy as np

ANN = "http://ann-benchmarks.com"
MSMARCO = "https://msmarco.z22.web.core.windows.net/msmarcoranking"

# Checksums of the files as downloaded on 2026-11-28. A mismatch stops the
# run unless --allow-checksum-mismatch is given (then it is recorded).
KNOWN = {
  "sift.tar.gz": (
    168280445,
    "92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a"),
  "glove-100-angular.hdf5": (
    485413888,
    "544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4"),
  "nytimes-256-angular.hdf5": (
    315208288,
    "6ad5d2cde6d6a6f21528d541901304ebfd40885d6ce6afae37bccbe278254a93"),
  "collection.tar.gz": (
    1035009698,
    "70667529e474322327d6441c8f7b621f2a805a7f6220897d9f80e5a8294bd62e"),
  "queries.tar.gz": (
    18882551,
    "05e4c62c9c8520cd695725340d4c990627fd1a92eb8a71b88e3c746031ba6de8"),
  "qrels.dev.small.tsv": (
    143300,
    "ddc607dbe7b748d7168e8675a242f0811d47febaa2e69d3af09d7ab249c68d7e"),
}

def log(msg):
  print(f"[prepare] {msg}", file=sys.stderr, flush=True)

def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while True:
      b = f.read(1 << 22)
      if not b:
        return h.hexdigest()
      h.update(b)

# Some hosts (ann-benchmarks.com since late 2026) answer 403 to urllib's
# default "Python-urllib/3.x" User-Agent, so send a neutral one.
USER_AGENT = "dataset-downloader/1.0"

def download(url, dest, args):
  """Fetches url to dest (unless present) and verifies it.

  Returns a source record for meta.json.
  """
  name = os.path.basename(dest)
  os.makedirs(os.path.dirname(dest), exist_ok=True)
  if not os.path.exists(dest):
    log(f"downloading {url}")
    tmp = dest + ".part"
    for attempt in range(3):
      try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r, \
            open(tmp, "wb") as f:
          shutil.copyfileobj(r, f, 1 << 22)
        break
      except OSError as e:
        log(f"  attempt {attempt + 1} failed: {e}")
        if attempt == 2:
          raise
        time.sleep(5)
    os.replace(tmp, dest)
  size = os.path.getsize(dest)
  digest = sha256(dest)
  rec = {"url": url, "file": name, "bytes": size, "sha256": digest}
  if name in KNOWN:
    ok = KNOWN[name] == (size, digest)
    rec["verified"] = ok
    if not ok:
      msg = (f"{dest}: size/sha256 {size}/{digest} differs from the "
             f"recorded {KNOWN[name][0]}/{KNOWN[name][1]}")
      if not args.allow_checksum_mismatch:
        raise SystemExit(msg + " (use --allow-checksum-mismatch to go on)")
      log("WARNING: " + msg)
  else:
    rec["verified"] = None
  return rec

def write_bin(path, a, dtype):
  a = np.ascontiguousarray(a, dtype=dtype)
  with open(path, "wb") as f:
    np.array(a.shape, dtype="<u4").tofile(f)
    a.astype(np.dtype(dtype).newbyteorder("<"), copy=False).tofile(f)

def read_vecs(path, dtype):
  """Reads TEXMEX .fvecs/.ivecs: per row an int32 d, then d values."""
  raw = np.fromfile(path, dtype="<i4")
  d = int(raw[0])
  rows = raw.reshape(-1, d + 1)
  if not (rows[:, 0] == d).all():
    raise SystemExit(f"{path}: inconsistent row dimensions")
  return rows[:, 1:].copy().view(dtype)

def file_record(path):
  return {"bytes": os.path.getsize(path), "sha256": sha256(path)}

def finish_vectors(out, name, metric, base, query, gt, sources, extra):
  os.makedirs(out, exist_ok=True)
  write_bin(os.path.join(out, "base.fbin"), base, "<f4")
  write_bin(os.path.join(out, "query.fbin"), query, "<f4")
  if gt.min() < 0 or gt.max() >= base.shape[0]:
    raise SystemExit(f"{name}: ground truth ids out of range")
  write_bin(os.path.join(out, "gt.ibin"), gt, "<i4")
  meta = {
    "name": name,
    "metric": metric,
    "n": int(base.shape[0]),
    "dim": int(base.shape[1]),
    "queries": int(query.shape[0]),
    "gt_k": int(gt.shape[1]),
    "sources": sources,
    "files": {f: file_record(os.path.join(out, f))
              for f in ("base.fbin", "query.fbin", "gt.ibin")},
    "prepared": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  }
  meta.update(extra)
  with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
  log(f"{name}: {meta['n']} x {meta['dim']} ({metric}), "
      f"{meta['queries']} queries, gt k={meta['gt_k']}")

def prepared(out):
  meta = os.path.join(out, "meta.json")
  if not os.path.exists(meta):
    return False
  with open(meta) as f:
    m = json.load(f)
  for name, rec in m.get("files", {}).items():
    p = os.path.join(out, name)
    if not os.path.exists(p) or os.path.getsize(p) != rec["bytes"]:
      return False
  return True

def from_hdf5(path, out, name, sources, extra):
  import h5py  # Only needed for HDF5 sources.
  with h5py.File(path, "r") as f:
    dist = f.attrs["distance"]
    dist = dist.decode() if isinstance(dist, bytes) else str(dist)
    metric = {"euclidean": "l2", "angular": "cosine", "dot": "dot"}[dist]
    base = np.asarray(f["train"], dtype=np.float32)
    query = np.asarray(f["test"], dtype=np.float32)
    gt = np.asarray(f["neighbors"], dtype=np.int64)
  extra = dict(extra, source_distance=dist)
  finish_vectors(out, name, metric, base, query, gt, sources, extra)

def prep_sift(args):
  name = "sift-128-euclidean"
  out = os.path.join(args.root, name)
  if prepared(out):
    log(f"{name}: already prepared")
    return
  raw = os.path.join(args.root, "raw", "sift")
  try:
    src = download("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
                   os.path.join(raw, "sift.tar.gz"), args)
  except (OSError, SystemExit) as e:
    log(f"TEXMEX FTP failed ({e}); using the ann-benchmarks HDF5 instead")
    src = download(f"{ANN}/{name}.hdf5", os.path.join(raw, f"{name}.hdf5"),
                   args)
    from_hdf5(os.path.join(raw, f"{name}.hdf5"), out, name, [src],
              {"origin": "ann-benchmarks (TEXMEX SIFT1M)"})
    return
  wanted = {f"sift/sift_{p}" for p in ("base.fvecs", "query.fvecs",
                                       "groundtruth.ivecs")}
  with tarfile.open(os.path.join(raw, "sift.tar.gz")) as t:
    for m in t.getmembers():
      if m.name in wanted and m.isfile():
        dest = os.path.join(raw, os.path.basename(m.name))
        with t.extractfile(m) as r, open(dest, "wb") as w:
          shutil.copyfileobj(r, w, 1 << 22)
  base = read_vecs(os.path.join(raw, "sift_base.fvecs"), np.float32)
  query = read_vecs(os.path.join(raw, "sift_query.fvecs"), np.float32)
  gt = read_vecs(os.path.join(raw, "sift_groundtruth.ivecs"), np.int32)
  finish_vectors(out, name, "l2", base, query, gt, [src],
                 {"origin": "TEXMEX SIFT1M (Jegou et al.)"})
  for f in ("sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"):
    os.remove(os.path.join(raw, f))

def prep_ann(args, name):
  out = os.path.join(args.root, name)
  if prepared(out):
    log(f"{name}: already prepared")
    return
  raw = os.path.join(args.root, "raw", name)
  path = os.path.join(raw, f"{name}.hdf5")
  src = download(f"{ANN}/{name}.hdf5", path, args)
  from_hdf5(path, out, name, [src], {"origin": "ann-benchmarks"})

def extract_member(tar_path, member, dest):
  """Copies one named regular file out of a tar archive."""
  with tarfile.open(tar_path) as t:
    m = t.getmember(member)
    if not m.isfile():
      raise SystemExit(f"{tar_path}: {member} is not a regular file")
    tmp = dest + ".part"
    with t.extractfile(m) as r, open(tmp, "wb") as w:
      shutil.copyfileobj(r, w, 1 << 22)
    os.replace(tmp, dest)

def prep_msmarco(args):
  name = "msmarco-passage"
  out = os.path.join(args.root, name)
  if prepared(out):
    log(f"{name}: already prepared")
    return
  os.makedirs(out, exist_ok=True)
  raw = os.path.join(args.root, "raw", "msmarco")
  srcs = [download(f"{MSMARCO}/{f}", os.path.join(raw, f), args)
          for f in ("collection.tar.gz", "queries.tar.gz",
                    "qrels.dev.small.tsv")]
  coll = os.path.join(out, "collection.tsv")
  log("extracting collection.tsv")
  extract_member(os.path.join(raw, "collection.tar.gz"), "collection.tsv",
                 coll)
  dev = os.path.join(raw, "queries.dev.tsv")
  extract_member(os.path.join(raw, "queries.tar.gz"), "queries.dev.tsv", dev)
  with open(os.path.join(raw, "qrels.dev.small.tsv")) as f:
    qids = {line.split("\t", 1)[0] for line in f if line.strip()}
  small = os.path.join(out, "queries.dev.small.tsv")
  kept = 0
  with open(dev, encoding="utf-8") as r, \
      open(small, "w", encoding="utf-8") as w:
    for line in r:
      if line.split("\t", 1)[0] in qids:
        w.write(line)
        kept += 1
  os.remove(dev)
  if kept != len(qids):
    raise SystemExit(f"found {kept} of {len(qids)} dev.small queries")
  # Passage ids must be 0..n-1 in order: the harness uses the line
  # number as the document id.
  n = 0
  with open(coll, encoding="utf-8") as f:
    for n, line in enumerate(f):
      if int(line.split("\t", 1)[0]) != n:
        raise SystemExit(f"collection.tsv line {n + 1}: unexpected id")
  meta = {
    "name": name,
    "passages": n + 1,
    "queries": kept,
    "sources": srcs,
    "files": {f: file_record(os.path.join(out, f))
              for f in ("collection.tsv", "queries.dev.small.tsv")},
    "prepared": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  }
  with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
  log(f"{name}: {n + 1} passages, {kept} queries")

def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--root", default=os.path.expanduser("~/datasets"))
  ap.add_argument("--datasets",
                  default="sift-128-euclidean,glove-100-angular,"
                  "msmarco-passage",
                  help="comma-separated; also nytimes-256-angular")
  ap.add_argument("--allow-checksum-mismatch", action="store_true")
  args = ap.parse_args()
  for d in filter(None, args.datasets.split(",")):
    if d == "sift-128-euclidean":
      prep_sift(args)
    elif d in ("glove-100-angular", "nytimes-256-angular"):
      prep_ann(args, d)
    elif d == "msmarco-passage":
      prep_msmarco(args)
    else:
      raise SystemExit(f"unknown dataset {d}")

if __name__ == "__main__":
  main()
