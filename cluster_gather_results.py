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
import pandas as pd


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
                if cand_pca.startswith("pca") or cand_pca == "raw":
                    inferred_pca = cand_pca
                if cand_method in ("kmeans", "agglomerative"):
                    inferred_method = cand_method
                if cand_source in ("actions", "images"):
                    inferred_source = inferred_source or cand_source
            # 3-level fallback: .../pca_or_raw/method/metrics_*.json
            elif len(parts) >= 3:
                cand_method = parts[-2]
                cand_pca = parts[-3]
                if cand_pca.startswith("pca") or cand_pca == "raw":
                    inferred_pca = cand_pca
                if cand_method in ("kmeans", "agglomerative"):
                    inferred_method = cand_method

            feature_source = inferred_source or "actions"

            use_pca = d.get(
                "use_pca",
                True if (inferred_pca and inferred_pca.startswith("pca")) else None,
            )
            pca_tag = inferred_pca or ("pca" if use_pca else "raw")

            # Filter: only keep v256 s1 for fast tokenizer
            if d.get("representation") == "fast":
                if d.get("vocab_size") != 256 or d.get("scale") != 1:
                    continue
            elif d.get("representation") == "native":
                continue

            row = {
                "feature_source": feature_source,
                "representation": d.get("representation", "?"),
                "image_encoder": d.get("image_encoder"),
                "delta_patches": d.get("delta_patches"),
                "vocab_size": d.get("vocab_size"),
                "scale": d.get("scale"),
                "cluster_method": d.get("cluster_method", inferred_method or "kmeans"),
                "pca_tag": pca_tag,
                "pca_90_comps": d.get("pca", {}).get("variance_90_comps", "?"),
                "pca_95_comps": d.get("pca", {}).get("variance_95_comps", "?"),
                "pca_99_comps": d.get("pca", {}).get("variance_99_comps", "?"),
                "var_top2": d.get("pca", {}).get(
                    "variance_top2", d.get("pca", {}).get("variance_2d", float("nan"))
                ),
                "var_top5": d.get("pca", {}).get("variance_top5", float("nan")),
                "var_top10": d.get("pca", {}).get("variance_top10", float("nan")),
                "var_top25": d.get("pca", {}).get("variance_top25", float("nan")),
                "var_top50": d.get("pca", {}).get("variance_top50", float("nan")),
                "ari": d.get("clustering", {}).get("ari", float("nan")),
                "nmi": d.get("clustering", {}).get("nmi", float("nan")),
                "silhouette": d.get("clustering", {}).get("silhouette", float("nan")),
                "purity": d.get("clustering", {}).get("purity", float("nan")),
                "depth": len(parts),
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
        return "FAST"
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
        f"{'Representation':<22} "
        f"{'90%':>5} {'99%':>5}  "
        f"{'V@5':>6} {'V@25':>6} {'V@50':>6}  "
        f"{'ARI':>7} {'NMI':>7} {'Sil':>7} {'Purity':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{label(r):<22} "
            f"{str(r['pca_90_comps']):>5} "
            f"{str(r['pca_99_comps']):>5}  "
            f"{fmt_f(r['var_top5'], pct=True):>6} "
            f"{fmt_f(r['var_top25'], pct=True):>6} "
            f"{fmt_f(r['var_top50'], pct=True):>6}  "
            f"{fmt_f(r['ari']):>7} "
            f"{fmt_f(r['nmi']):>7} "
            f"{fmt_f(r['silhouette']):>7} "
            f"{fmt_f(r['purity']):>7}"
        )
    print(sep)
    print(f"  {len(rows)} result(s)\n")


def save_to_latex(rows, title, fs, pca_tag, method):
    """Generate LaTeX table for a single group."""
    df = pd.DataFrame(rows)
    df["representation_label"] = df.apply(label, axis=1)

    # Simplified column headers for LaTeX
    mapping = {
        "representation_label": "Rep",
        "pca_90_comps": "C@90\\%",
        "pca_99_comps": "C@99\\%",
        "var_top5": "Var@5",
        "var_top25": "Var@25",
        "var_top50": "Var@50",
        "ari": "ARI",
        "nmi": "NMI",
        "silhouette": "Sil",
        "purity": "Pur",
    }

    cols = [c for c in mapping.keys() if c in df.columns]
    df = df[cols].copy()

    # Format percentages and floats
    for c in df.columns:
        if c.startswith("var_top"):
            df[c] = df[c].apply(
                lambda x: f"{x:.1%}".replace("%", "\\%") if x == x else "---"
            )
        elif c in ("ari", "nmi", "silhouette", "purity"):
            df[c] = df[c].apply(lambda x: f"{x:.2f}" if x == x else "---")

    df = df.rename(columns=mapping)

    # Create description
    source_label = "actions" if fs == "actions" else "images"

    pca_desc = ""
    if pca_tag == "pca":
        pca_desc = "intrinsic subspace (components explaining 99\\% variance)"
    elif pca_tag == "pca10":
        pca_desc = "fixed projection to 10 principal components"
    elif pca_tag == "pca50":
        pca_desc = "fixed projection to 50 principal components"
    elif pca_tag == "raw":
        pca_desc = "raw scaled features"
    else:
        pca_desc = f"{pca_tag} features"

    caption = f"Clustering results for {source_label} using {method} on {pca_desc}."

    tex = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:clustering_" + f"{fs}_{pca_tag}_{method}".replace("@", "_") + "}",
        df.to_latex(index=False, escape=False),
        "\\end{table}",
        "\n",
    ]
    return "\n".join(tex)


def save_to_csv(rows, out_path):
    """Save rows to CSV using pandas for clean formatting."""
    df = pd.DataFrame(rows)

    # Apply labeling to representation column BEFORE filtering
    df["representation_label"] = df.apply(label, axis=1)

    # Reorder/rename columns for clarity
    cols = [
        "representation_label",
        "pca_90_comps",
        "pca_99_comps",
        "var_top5",
        "var_top25",
        "var_top50",
        "ari",
        "nmi",
        "silhouette",
        "purity",
    ]
    # Filter to actual columns present
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df = df.rename(columns={"representation_label": "representation"})

    df.to_csv(out_path, index=False)
    print(f"  [CSV] Saved to {out_path}")


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
        # Keep deeper path (more structured) or later file if same depth
        if key not in seen or r["depth"] >= seen[key]["depth"]:
            seen[key] = r
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results/clustering")
    parser.add_argument(
        "--csv_dir", default=None, help="Directory to save CSV summaries"
    )
    parser.add_argument(
        "--tex_file", default=None, help="Filename to save LaTeX tables"
    )
    args = parser.parse_args()

    print(f"\nScanning {args.results_dir} ...\n")
    rows = load_metrics(args.results_dir)

    if not rows:
        print("No results found yet.")
        return

    rows = dedup(rows)

    groups = sorted(
        set((r["feature_source"], r["pca_tag"], r["cluster_method"]) for r in rows)
    )
    tex_content = []

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
        group_title = f"{source_label} | {pca_tag} | {method}"
        print_table(group, title=group_title)

        if args.csv_dir:
            os.makedirs(args.csv_dir, exist_ok=True)
            csv_name = f"summary_{fs}_{pca_tag}_{method}.csv"
            save_to_csv(group, os.path.join(args.csv_dir, csv_name))

        if args.tex_file:
            tex_content.append(save_to_latex(group, group_title, fs, pca_tag, method))

    if args.tex_file:
        with open(args.tex_file, "w") as f:
            f.write("%% Auto-generated clustering tables\n")
            f.write("\n".join(tex_content))
        print(f"  [TEX] Saved to {args.tex_file}")

    print(
        f"Total: {len(rows)} unique result(s) across {len(groups)} configuration(s).\n"
    )


if __name__ == "__main__":
    main()
