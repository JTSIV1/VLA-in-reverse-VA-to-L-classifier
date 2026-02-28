#!/usr/bin/env python3
"""
Gather cluster analysis results from results/clustering/<pca|raw>/<method>/metrics_*.json
Safe to run while jobs are running — prints whatever is available.
Results are grouped into separate tables by (feature_space, cluster_method).

Usage:
    python cluster_gather_results.py [--results_dir ./results/clustering]
"""

import argparse
import json
import os
import glob


def load_metrics(results_dir):
    # Scan nested structure: <base>/<pca_or_raw>/<method>/metrics_*.json
    # Also scan flat legacy: <base>/metrics_*.json for backward compat
    patterns = [
        os.path.join(results_dir, "*", "*", "metrics_*.json"),
        os.path.join(results_dir, "metrics_*.json"),
    ]
    files = sorted(set(f for pat in patterns for f in glob.glob(pat)))
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)

            # Infer pca_tag and cluster_method from directory if not in JSON
            parts = os.path.normpath(f).split(os.sep)
            inferred_pca = None
            inferred_method = None
            if len(parts) >= 3:
                candidate_method = parts[-2]
                candidate_pca = parts[-3]
                if candidate_pca in ("pca", "raw"):
                    inferred_pca = candidate_pca
                if candidate_method in ("kmeans", "agglomerative"):
                    inferred_method = candidate_method

            use_pca = d.get("use_pca", True if inferred_pca == "pca" else None)
            pca_tag = (
                "pca"
                if use_pca
                else ("raw" if use_pca is False else (inferred_pca or "?"))
            )

            row = {
                "representation": d.get("representation", "?"),
                "vocab_size": d.get("vocab_size"),
                "scale": d.get("scale"),
                "cluster_method": d.get("cluster_method", inferred_method or "kmeans"),
                "pca_tag": pca_tag,
                "pca_99_comps": d.get("pca", {}).get("variance_99_comps", "?"),
                "pca_2d_var": d.get("pca", {}).get("variance_2d", float("nan")),
                "ari": d.get("clustering", {}).get("ari", float("nan")),
                "nmi": d.get("clustering", {}).get("nmi", float("nan")),
                "silhouette": d.get("clustering", {}).get("silhouette", float("nan")),
                "purity": d.get("clustering", {}).get("purity", float("nan")),
                "_file": f,
            }
            rows.append(row)
        except Exception as e:
            print(f"  [WARN] Could not read {f}: {e}")
    return rows


def fmt_f(v, pct=False):
    if v is None or v != v:
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
        print(f"\n{'━' * 74}")
        print(f"  {title}")
        print(f"{'━' * 74}")

    header = (
        f"{'Tokenizer':<20}  {'99%PCA':>7}  {'2D var':>7}  "
        f"{'ARI':>7}  {'NMI':>7}  {'Sil':>7}  {'Purity':>7}"
    )
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


_ORDER = {"native": 0, "bin": 1, "quest": 2, "oat": 3, "fast": 4}


def sort_key(r):
    rep = r["representation"]
    base = _ORDER.get(rep, 5)
    if rep == "fast":
        return (base, r["vocab_size"] or 0, r["scale"] or 0)
    return (base, 0, 0)


def dedup(rows):
    """Drop duplicate (representation, vocab, scale, cluster_method, pca_tag) keeping last file."""
    seen = {}
    for r in rows:
        key = (
            r["representation"],
            r["vocab_size"],
            r["scale"],
            r["cluster_method"],
            r["pca_tag"],
        )
        seen[key] = r
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results/clustering")
    args = parser.parse_args()

    print(f"\nScanning {args.results_dir} ...\n")
    rows = load_metrics(args.results_dir)

    if not rows:
        print("No results found yet.")
        return

    rows = dedup(rows)

    # Group: pca_tag first (pca before raw), then cluster_method alphabetically
    groups = sorted(set((r["pca_tag"], r["cluster_method"]) for r in rows))
    for pca_tag, method in groups:
        group = sorted(
            [
                r
                for r in rows
                if r["pca_tag"] == pca_tag and r["cluster_method"] == method
            ],
            key=sort_key,
        )
        print_table(group, title=f"Features: {pca_tag}  |  Clustering: {method}")

    print(
        f"Total: {len(rows)} unique result(s) across {len(groups)} configuration(s).\n"
    )


if __name__ == "__main__":
    main()
