#!/usr/bin/env python3
"""Turns a directory of result JSON files into a Markdown report.

usage: report.py RESULTS_DIR > REPORT.md
"""

import glob
import json
import os
import sys
import time

RECALL_TARGETS = [0.80, 0.90, 0.95, 0.98, 0.99]

def load(d):
  out = []
  for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    with open(p) as f:
      try:
        r = json.load(f)
      except json.JSONDecodeError:
        continue
    if isinstance(r, dict) and "kind" in r:
      r["_file"] = os.path.basename(p)
      out.append(r)
  return out

def mb(b):
  return "-" if b is None else f"{b / 1e6:,.1f}"

def num(x, fmt="{:,.0f}"):
  return "-" if x is None else fmt.format(x)

def table(head, rows):
  lines = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
  lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
  return "\n".join(lines)

def params_str(r):
  p = r.get("params", {})
  if p.get("exact"):
    return "exact"
  m = p.get("M", p.get("connectivity"))
  efc = p.get("ef_construction", p.get("expansion_add"))
  s = f"M={m} efC={efc}"
  q = p.get("quantization")
  if q and q != "None":
    s += f" {q} rerank={p.get('rerank')}"
  return s

def pareto(points):
  """Points not dominated in (recall, qps_1t)."""
  pts = sorted(points, key=lambda p: (-p["recall"], -p["qps_1t"]))
  out, best = [], -1.0
  for p in pts:
    if p["qps_1t"] > best:
      out.append(p)
      best = p["qps_1t"]
  return sorted(out, key=lambda p: p["recall"])

def vector_section(ds, runs, checks):
  r0 = runs[0]["dataset"]
  lines = [f"## {ds}", "",
           f"{r0['n']:,} vectors, {r0['dim']} dimensions, metric "
           f"`{r0['metric']}`, {r0['queries']:,} queries, k = "
           f"{runs[0]['k']}, {runs[0]['threads']} threads.", ""]
  for c in checks:
    lines.append(
      f"Ground truth check: muninn exact search on {c['sampled']} sampled "
      f"queries matches the provided neighbours for "
      f"{c['id_set_match'] * 100:.1f}% of queries by id, "
      f"{c['match_with_ties'] * 100:.1f}% counting distance ties "
      f"(recall {c['recall']:.4f}).")
    lines.append("")
  lines += ["### Build", ""]
  rows = []
  for r in runs:
    b = r["build"]
    rows.append([r["engine"], r["engine_version"], params_str(r),
                 num(b["seconds"], "{:,.1f}"), b["threads"],
                 mb(b["index_bytes"]), mb(b.get("rss_delta_bytes"))])
  lines += [table(["engine", "version", "params", "build s", "threads",
                   "index MB", "RSS growth MB"], rows), ""]
  lines += ["### Best QPS at a recall target (one thread / all threads)",
            "", "Highest throughput among sweep points reaching at least "
            "the given recall@k; `ef` in brackets.", ""]
  rows = []
  for r in runs:
    if r["params"].get("exact"):
      continue
    row = [r["engine"] + " " + params_str(r)]
    for t in RECALL_TARGETS:
      ok = [p for p in r["results"] if p["recall"] >= t]
      if not ok:
        row.append("-")
        continue
      b1 = max(ok, key=lambda p: p["qps_1t"])
      bn = max(ok, key=lambda p: p["qps_nt"])
      row.append(f"{b1['qps_1t']:,.0f} ({b1['ef']}) / "
                 f"{bn['qps_nt']:,.0f} ({bn['ef']})")
    rows.append(row)
  lines += [table(["engine"] + [f"recall ≥ {t}" for t in RECALL_TARGETS],
                  rows), ""]
  for r in runs:
    lines += [f"### {r['engine']} {params_str(r)}", ""]
    front = {id(p) for p in pareto(r["results"])}
    rows = []
    for p in r["results"]:
      l = p["latency"]
      rows.append([p["ef"] or "-", f"{p['recall']:.4f}",
                   num(p["qps_1t"]), num(p["qps_nt"]),
                   num(l["p50_us"], "{:,.1f}"), num(l["p95_us"], "{:,.1f}"),
                   num(l["p99_us"], "{:,.1f}"),
                   "yes" if id(p) in front else ""])
    lines += [table(["ef", "recall@k", "QPS 1t",
                     f"QPS {r['threads']}t", "p50 us", "p95 us", "p99 us",
                     "Pareto"], rows), ""]
    if r["engine"] in ("hnswlib", "faiss-hnsw", "usearch"):
      ov = [p.get("python_call_overhead_us") for p in r["results"]]
      ov = [o for o in ov if o is not None]
      if ov:
        lines += [f"Latency columns include the Python call; per-query "
                  f"overhead versus the batch call was "
                  f"{min(ov):.1f}..{max(ov):.1f} us.", ""]
  return lines

def text_section(runs):
  lines = ["## MS MARCO passages, BM25 top-k", ""]
  for r in sorted(runs, key=lambda r: r["analyzer"]):
    d = r["dataset"]
    lines += [f"### Analyzer `{r['analyzer']}`", "",
              f"{d['docs']:,} passages ({d['text_bytes'] / 1e6:,.0f} MB of "
              f"text), {d['queries']:,} queries "
              f"({d.get('queries_file')}), k = {r['k']}, "
              f"{r['threads']} query threads.", ""]
    rows = []
    for e in r["engines"]:
      i, s = e["index"], e["search"]
      l = s["latency"]
      rows.append([e["engine"], e["engine_version"], i["threads"],
                   num(i["total_seconds"], "{:,.1f}"),
                   num(i["docs_per_sec"]), mb(i["index_bytes"]),
                   num(s["qps_1t"]), num(s["qps_nt"]),
                   num(l["p50_us"], "{:,.1f}"), num(l["p95_us"], "{:,.1f}"),
                   num(l["p99_us"], "{:,.1f}"),
                   num(s["mean_hits"], "{:.2f}")])
    lines += [table(["engine", "version", "index threads", "index s",
                     "docs/s", "index MB", "QPS 1t", f"QPS {r['threads']}t",
                     "p50 us", "p95 us", "p99 us", "hits/query"], rows), ""]
    for e in r["engines"]:
      for o in e.get("other_index_runs", []):
        lines.append(
          f"- {e['engine']} indexing with {o['threads']} threads: "
          f"{o['total_seconds']:,.1f} s ({o['docs_per_sec']:,.0f} docs/s; "
          f"{o['index_seconds']:,.1f} s indexing + "
          f"{o['finish_seconds']:,.1f} s merging), "
          f"{mb(o['index_bytes'])} MB")
      i = e["index"]
      lines.append(
        f"- {e['engine']}: {i['index_seconds']:,.1f} s indexing + "
        f"{i['finish_seconds']:,.1f} s "
        f"{'writing' if e['engine'] == 'muninn' else 'merging'}; analyzer: "
        f"{e['analyzer']}")
      fast = i.get("details", {}).get("id_fast_field_bytes")
      if fast:
        lines.append(f"- tantivy index size includes {mb(fast)} MB for the "
                     f"passage-number fast field used to map results.")
    o = r.get("overlap")
    if o:
      lines.append(
        f"- Result agreement {o['engines'][0]} vs {o['engines'][1]}: mean "
        f"overlap@{r['k']} {o['mean']:.4f}, identical top-{r['k']} sets "
        f"{o['identical_sets'] * 100:.1f}%, same top-1 "
        f"{o['same_top1'] * 100:.1f}%.")
    lines.append("")
  return lines

def main():
  d = sys.argv[1]
  rs = load(d)
  if not rs:
    raise SystemExit(f"no results in {d}")
  # system.json comes from the Rust binary and has every field.
  sysinfo = next((r["system"] for r in rs if "system" in r), {})
  sys_path = os.path.join(d, "system.json")
  if os.path.exists(sys_path):
    with open(sys_path) as f:
      sysinfo = json.load(f)
  lines = ["# muninn benchmark report", ""]
  if sysinfo.get("label"):
    lines += [f"**{sysinfo['label']}**", ""]
  lines += [
    f"- Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
    f"- Machine: {sysinfo.get('machine_type') or 'unknown'}, "
    f"{sysinfo.get('cpu_model')}, {sysinfo.get('available_cpus')} CPUs "
    f"available (pinned to {sysinfo.get('cpu_pinning') or 'all'}), "
    f"{mb(sysinfo.get('mem_total_bytes'))} MB RAM",
    f"- CPU flags: {', '.join(sysinfo.get('cpu_flags', []))}",
    f"- muninn commit {sysinfo.get('git_commit') or 'unknown'}, "
    f"{sysinfo.get('rustc') or 'rustc unknown'}, RUSTFLAGS "
    f"`{sysinfo.get('rustflags') or ''}`, SIMD "
    f"{sysinfo.get('muninn_simd_isa', '?')}",
    "",
  ]
  versions = sorted({(r["engine"], r["engine_version"])
                     for r in rs if r["kind"] == "vector"} |
                    {(e["engine"], e["engine_version"])
                     for r in rs if r["kind"] == "text"
                     for e in r["engines"]})
  lines += ["Engines: " + ", ".join(f"{e} {v}" for e, v in versions), ""]
  vec = [r for r in rs if r["kind"] == "vector"]
  checks = [r for r in rs if r["kind"] == "gt-check"]
  order = {"muninn-hnsw": 0, "muninn-hnsw-sq8": 1, "muninn-hnsw-binary": 2,
           "hnswlib": 3, "faiss-hnsw": 4, "usearch": 5, "muninn-flat": 9}
  for ds in sorted({r["dataset"]["name"] for r in vec}):
    runs = sorted((r for r in vec if r["dataset"]["name"] == ds),
                  key=lambda r: (order.get(r["engine"], 8), params_str(r)))
    lines += vector_section(ds, runs, [c for c in checks
                                       if c["dataset"] == ds])
  text = [r for r in rs if r["kind"] == "text"]
  if text:
    lines += text_section(text)
  print("\n".join(lines))

if __name__ == "__main__":
  main()
