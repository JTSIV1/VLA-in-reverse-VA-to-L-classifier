#!/usr/bin/env python3
"""
Gather cluster analysis results from results/clustering/metrics_*.json and print summary tables.
Safe to run while jobs are still running — prints whatever is available.
Results are grouped and printed in a separate table per clustering method.

Usage:
    python cluster_gather_results.py [--results_dir ./results/clustering]
"""

import argparse
import json
import os
import glob


def load_metrics(results_dir):
    pattern = os.path.join(results_dir, "metrics_*.json")
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            row = {
                "representation": d.get("representation", "?"),
                "vocab_size": d.get("vocab_size"),
                "scale": d.get("scale"),
                "cluster_method": d.get(
                    "cluster_method", "kmeans"
                ),  # legacy files default to kmeans
                "pca_99_comps": d.get("pca", {}).get("variance_99_comps", "?"),
                "pca_2d_var": d.get("pca", {}).get("variance_2d", float("nan")),
                "ari": d.get("clustering", {}).get("ari", float("nan")),
                "nmi": d.get("clustering", {}).get("nmi", float("nan")),
                "silhouette": d.get("clustering", {}).get("silhouette", float("nan")),
                "purity": d.get("clustering", {}).get("purity", float("nan")),
                "_file": os.path.basename(f),
            }
            rows.append(row)
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")
    return rows


def fmt_f(v, pct=False):
    if v is None or v != v:  # None or NaN
        return "   —   "
    if pct:
        return f"{v:.1%}"
    return f"{v:.3f}"


def label(r):
    rep = r["representation"]
    if rep == "fast":
        v = r["vocab_size"] or "?"
        s = (
            int(r["scale"])
            if r["scale"] is not None and r["scale"] == int(r["scale"])
            else r["scale"]
        )
        return f"fast v{v} s{s}"
    return rep


def print_table(rows, title=None):
    if not rows:
        print("  (no results)")
        return

    if title:
        print(f"\n{'━' * 72}")
        print(f"  {title}")
        print(f"{'━' * 72}")

    header = f"{'Tokenizer':<20}  {'99%PCA':>7}  {'2D var':>7}  {'ARI':>7}  {'NMI':>7}  {'Sil':>7}  {'Purity':>7}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{label(r):<20}  "
            f"{str(r['pca_99_comps']):>7}  "
            f"{fmt_f(r['pca_2d_var'], pct=True):>7}  "
            f"{fmt_f(r['ari']):>7}  "
            f"{fmt_f(r['nmi']):>7}  "
            f"{fmt_f(r['silhouette']):>7}  "
            f"{fmt_f(r['purity']):>7}"
        )
    print(sep)
    print(f"  {len(rows)} result(s)\n")


# Sort: native, bin, quest, oat, then fast by vocab then scale
_ORDER = {"native": 0, "bin": 1, "quest": 2, "oat": 3, "fast": 4}


def sort_key(r):
    rep = r["representation"]
    base = _ORDER.get(rep, 5)
    if rep == "fast":
        return (base, r["vocab_size"] or 0, r["scale"] or 0)
    return (base, 0, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results/clustering")
    args = parser.parse_args()

    print(f"\nScanning {args.results_dir} ...\n")
    rows = load_metrics(args.results_dir)

    if not rows:
        print("No results found yet.")
        return

    # Group by cluster_method, print a separate table for each
    methods = sorted(set(r["cluster_method"] for r in rows))
    for method in methods:
        group = sorted([r for r in rows if r["cluster_method"] == method], key=sort_key)
        print_table(group, title=f"Clustering method: {method}")

    print(f"Total: {len(rows)} result(s) across {len(methods)} method(s).\n")


if __name__ == "__main__":
    main()
