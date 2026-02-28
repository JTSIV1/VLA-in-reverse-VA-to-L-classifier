#!/usr/bin/env python3
"""
Gather cluster analysis results from
    results/clustering/<feature_source>/<pca|raw>/<method>/metrics_*.json

Safe to run while jobs are running — prints whatever is available.
Results are grouped into separate tables by (feature_source, pca_tag, cluster_method).

Usage:
    python cluster_gather_results.py [--results_dir ./results/clustering]
"""

import argparse
import json
import os
import glob


def load_metrics(results_dir):
    # Scan nested structure (4-level): <base>/<feature_source>/<pca_or_raw>/<method>/metrics_*.json
    # Also scan 3-level (legacy):      <base>/<pca_or_raw>/<method>/metrics_*.json
    # Also scan flat legacy:           <base>/metrics_*.json
    patterns = [
        os.path.join(results_dir, "*", "*", "*", "metrics_*.json"),
        os.path.join(results_dir, "*", "*", "metrics_*.json"),
        os.path.join(results_dir, "metrics_*.json"),
    ]
    files = sorted(set(f for pat in patterns for f in glob.glob(pat)))
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)

            # Infer structure from directory path
            parts = os.path.normpath(f).split(os.sep)

            # Try to detect feature_source / pca_tag / method from path
            inferred_source = d.get("feature_source")
            inferred_pca = None
            inferred_method = None

            # 4-level: .../feature_source/pca_or_raw/method/metrics_*.json
            if len(parts) >= 4:
                cand_method = parts[-2]
                cand_pca = parts[-3]
                cand_source = parts[-4]
                if cand_pca in ("pca", "raw"):
                    inferred_pca = cand_pca
                if cand_method in ("kmeans", "agglomerative"):
                    inferred_method = cand_method
                if cand_source in ("actions", "images"):
                    inferred_source = inferred_source or cand_source
            # 3-level fallback: .../pca_or_raw/method/metrics_*.json
            elif len(parts) >= 3:
                cand_method = parts[-2]
                cand_pca = parts[-3]
                if cand_pca in ("pca", "raw"):
                    inferred_pca = cand_pca
                if cand_method in ("kmeans", "agglomerative"):
                    inferred_method = cand_method

            feature_source = inferred_source or "actions"

            use_pca = d.get("use_pca", True if inferred_pca == "pca" else None)
            pca_tag = (
                "pca"
                if use_pca
                else ("raw" if use_pca is False else (inferred_pca or "?"))
            )

            row = {
                "feature_source": feature_source,
                "representation": d.get("representation", "?"),
                "image_encoder": d.get("image_encoder"),
                "delta_patches": d.get("delta_patches"),
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
    fs = r["feature_source"]

    if fs == "images":
        enc = r.get("image_encoder") or rep
        dp = r.get("delta_patches")
        if dp and dp > 0:
            return f"{enc} (Δ{dp})"
        return f"{enc} (full)"

    # Action representations
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
        print(f"\n{'━' * 78}")
        print(f"  {title}")
        print(f"{'━' * 78}")

    header = (
        f"{'Representation':<22}  {'99%PCA':>7}  {'2D var':>7}  "
        f"{'ARI':>7}  {'NMI':>7}  {'Sil':>7}  {'Purity':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{label(r):<22}  "
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
_IMG_ORDER = {
    "resnet18": 0,
    "dinov2": 1,
    "dinov2_s": 2,
    "dinov2_b": 3,
    "vc1": 4,
    "r3m": 5,
}


def sort_key(r):
    fs = r["feature_source"]
    if fs == "images":
        enc = r.get("image_encoder") or r["representation"]
        return (0, _IMG_ORDER.get(enc, 10), r.get("delta_patches") or 0)
    rep = r["representation"]
    base = _ORDER.get(rep, 5)
    if rep == "fast":
        return (1, base, r["vocab_size"] or 0, r["scale"] or 0)
    return (1, base, 0, 0)


def dedup(rows):
    """Drop duplicate entries keeping last file."""
    seen = {}
    for r in rows:
        key = (
            r["feature_source"],
            r["representation"],
            r.get("image_encoder"),
            r.get("delta_patches"),
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

    # Group: feature_source first, then pca_tag, then cluster_method
    groups = sorted(
        set((r["feature_source"], r["pca_tag"], r["cluster_method"]) for r in rows)
    )
    for fs, pca_tag, method in groups:
        group = sorted(
            [
                r
                for r in rows
                if r["feature_source"] == fs
                and r["pca_tag"] == pca_tag
                and r["cluster_method"] == method
            ],
            key=sort_key,
        )
        source_label = "Actions" if fs == "actions" else "Images"
        print_table(
            group,
            title=f"{source_label}  |  Features: {pca_tag}  |  Clustering: {method}",
        )

    print(
        f"Total: {len(rows)} unique result(s) across {len(groups)} configuration(s).\n"
    )


if __name__ == "__main__":
    main()
