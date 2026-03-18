"""
Visualize hierarchical annotations: show subsampled frames with L1 phase labels.

Each episode gets its own figure panel with large frames, colored borders by phase,
and a legend showing phase descriptions with time ranges.

Usage:
  python scripts/visualize_hierarchy_annotations.py \
      --annotations data/hierarchy_annotations/calvin_training.jsonl \
      --max_episodes 10 --output figures/hierarchy_pilot.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR, VAL_DIR, IMAGE_KEY, EPISODE_TEMPLATE

# Distinguishable colors for phases
PHASE_COLORS = [
    "#2176AE", "#E8871E", "#3EA658", "#D04545", "#7B5EA7",
    "#B07D4B", "#D370B0", "#6B6B6B", "#B8A830", "#3DA0BD",
]


def load_frames(data_dir, start_idx, end_idx, frame_indices):
    """Load specific frames from CALVIN episodes."""
    frames = []
    for fi in frame_indices:
        global_idx = start_idx + fi
        ep_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(global_idx))
        ep = np.load(ep_path)
        frames.append(ep[IMAGE_KEY])
    return frames


def get_phase_for_frame(frame_idx, decomposition):
    """Return which phase index a frame timestep falls into."""
    for i, phase in enumerate(decomposition):
        if phase["START_TIMESTEP"] <= frame_idx <= phase["END_TIMESTEP"]:
            return i
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--output", default="figures/hierarchy_pilot.png")
    args = parser.parse_args()

    data_dir = TRAIN_DIR if args.split == "training" else VAL_DIR

    episodes = []
    with open(args.annotations) as f:
        for line in f:
            episodes.append(json.loads(line))
            if len(episodes) >= args.max_episodes:
                break

    n_eps = len(episodes)
    n_frames = max(len(ep["frame_indices"]) for ep in episodes)

    # Each episode: 1 row for frames + space for legend text
    # Use GridSpec: 2 rows per episode (frames row + legend row)
    fig_width = n_frames * 2.5
    row_height_frames = 2.5
    row_height_legend = 1.8
    fig_height = n_eps * (row_height_frames + row_height_legend) + 0.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        n_eps * 2, n_frames, figure=fig,
        height_ratios=[row_height_frames, row_height_legend] * n_eps,
        hspace=0.4, wspace=0.08,
    )

    for i, ep in enumerate(episodes):
        frame_indices = ep["frame_indices"]
        decomp = ep["decomposition"]
        n_f = len(frame_indices)

        frames = load_frames(
            data_dir, ep["start_idx"], ep["end_idx"], frame_indices
        )
        phase_ids = [get_phase_for_frame(fi, decomp) for fi in frame_indices]

        # -- Frame row --
        frame_row = i * 2
        for j in range(n_f):
            ax = fig.add_subplot(gs[frame_row, j])
            ax.imshow(frames[j])
            ax.set_xticks([])
            ax.set_yticks([])

            pid = phase_ids[j]
            color = PHASE_COLORS[pid % len(PHASE_COLORS)] if pid >= 0 else "#CCCCCC"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)

            ax.set_xlabel("t={}".format(frame_indices[j]), fontsize=9, labelpad=2)

            # Phase label above frame (only on first frame of each phase)
            if j == 0 or phase_ids[j] != phase_ids[j - 1]:
                ax.set_title(
                    decomp[pid]["STEP_DESCRIPTION"],
                    fontsize=9, color=color, fontweight="bold", pad=4,
                    ha="left", loc="left",
                )

        # Hide unused frame cells
        for j in range(n_f, n_frames):
            ax = fig.add_subplot(gs[frame_row, j])
            ax.set_visible(False)

        # -- Legend row (spans all columns) --
        legend_row = i * 2 + 1
        ax_leg = fig.add_subplot(gs[legend_row, :])
        ax_leg.axis("off")

        inst_gt = ep.get("instruction_gt", ep.get("instruction", ""))
        inst_gemini = ep.get("instruction_gemini", "")

        # Build legend text
        parts = []
        for pid, phase in enumerate(decomp):
            color = PHASE_COLORS[pid % len(PHASE_COLORS)]
            parts.append((
                "[t={}..{}] {}".format(
                    phase["START_TIMESTEP"], phase["END_TIMESTEP"],
                    phase["STEP_DESCRIPTION"],
                ),
                color,
            ))

        # Title: GT L0 and Gemini L0
        y_cursor = 1.0
        ax_leg.text(0.0, y_cursor, "ep{}".format(ep["episode_index"]),
                    fontsize=11, fontweight="bold",
                    transform=ax_leg.transAxes, va="top")
        ax_leg.text(0.04, y_cursor,
                    "  GT (L0): \"{}\"".format(inst_gt),
                    fontsize=10, fontweight="bold", color="#B00020",
                    transform=ax_leg.transAxes, va="top")
        if inst_gemini:
            y_cursor -= 0.18
            ax_leg.text(0.04, y_cursor,
                        "  Gemini (L0): \"{}\"".format(inst_gemini),
                        fontsize=10, fontweight="bold", color="#1565C0",
                        transform=ax_leg.transAxes, va="top")

        # Phase descriptions (L1)
        y_cursor -= 0.18
        ax_leg.text(0.0, y_cursor, "L1 phases:", fontsize=8, fontweight="bold",
                    color="#444444", transform=ax_leg.transAxes, va="top")
        for k, (text, color) in enumerate(parts):
            x = (k % 3) * 0.34
            y = y_cursor - 0.15 - (k // 3) * 0.28
            ax_leg.text(x, y, text, fontsize=8, color=color,
                        transform=ax_leg.transAxes, va="top",
                        fontweight="semibold")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved to {}".format(args.output))


if __name__ == "__main__":
    main()
