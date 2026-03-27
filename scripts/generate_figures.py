"""
Generate figures for the Building Systems module.

Produces:
  - docs/images/pipeline_architecture.png — 4-step pipeline diagram
  - docs/images/routing_diagram.png — 5-specialty router pattern
  - docs/images/moderation_results.png — Pass/block/flag bar chart
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# --- Theme ---
plt.style.use("dark_background")
BG_COLOR = "#1a1a2e"
COLORS = ["#4f7cac", "#5a9e8f", "#9b6b9e", "#c47e3a", "#b85450"]
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4e"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_pipeline_architecture():
    """4-step pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7, 5.5, "Clinical Triage Pipeline Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=TEXT_COLOR)

    # Pipeline steps
    steps = [
        ("Step 1\nExtraction", "Extract structured\ndata from note", COLORS[0], "gpt-4o-mini"),
        ("Step 2\nRed Flags", "Assess red flags\nand acuity", COLORS[1], "gpt-4o-mini"),
        ("Step 3\nUrgency", "Classify urgency\n1-5 scale", COLORS[2], "gpt-4o"),
        ("Step 4\nFormat", "Generate final\nstructured output", COLORS[3], "gpt-4o-mini"),
    ]

    x_positions = [1.5, 4.5, 7.5, 10.5]

    # Input arrow
    ax.annotate("", xy=(0.8, 3), xytext=(0, 3),
                arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))
    ax.text(0.05, 3.5, "Clinical\nNote", ha="left", va="center",
            fontsize=9, color="#888888")

    for i, (title, desc, color, model) in enumerate(steps):
        px = x_positions[i]

        # Box
        box = FancyBboxPatch((px - 1.2, 1.8), 2.4, 2.4,
                              boxstyle="round,pad=0.2", facecolor=color,
                              edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85)
        ax.add_patch(box)

        ax.text(px, 3.5, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(px, 2.7, desc, ha="center", va="center",
                fontsize=8, color="#e0e0e0", style="italic")
        ax.text(px, 2.1, f"[{model}]", ha="center", va="center",
                fontsize=7, color="#cccccc")

        # Arrow to next step
        if i < len(steps) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 1.2, 3),
                        xytext=(px + 1.2, 3),
                        arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))

    # Output arrow
    ax.annotate("", xy=(13.5, 3), xytext=(11.7, 3),
                arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))
    ax.text(13.6, 3.5, "Structured\nJSON", ha="left", va="center",
            fontsize=9, color="#888888")

    # Timing bar
    ax.text(7, 0.8, "Total Pipeline: ~3.2s  |  Each step validates before passing to next",
            ha="center", va="center", fontsize=9, color="#888888", style="italic")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pipeline_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'pipeline_architecture.png'}")


def fig_routing_diagram():
    """5-specialty router pattern diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.5, "Clinical Input Classification & Routing",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=TEXT_COLOR)

    # Input
    input_box = FancyBboxPatch((0.5, 3.5), 2.2, 1.2,
                                boxstyle="round,pad=0.15", facecolor="#2a2a4e",
                                edgecolor=TEXT_COLOR, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.6, 4.1, "Clinical\nNote Input", ha="center", va="center",
            fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # Classifier
    clf_box = FancyBboxPatch((3.8, 3.3), 2.4, 1.6,
                              boxstyle="round,pad=0.2", facecolor=COLORS[0],
                              edgecolor=TEXT_COLOR, linewidth=2, alpha=0.9)
    ax.add_patch(clf_box)
    ax.text(5, 4.4, "CLASSIFIER", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    ax.text(5, 3.8, "gpt-4o-mini\nRoute selection", ha="center", va="center",
            fontsize=8, color="#d0d0d0")

    # Arrow from input to classifier
    ax.annotate("", xy=(3.8, 4.1), xytext=(2.7, 4.1),
                arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))

    # Routes
    routes = [
        ("Cardiology", "Chest pain, arrhythmia,\nheart failure", COLORS[0], 6.2),
        ("Respiratory", "Dyspnea, COPD,\npneumonia", COLORS[1], 4.8),
        ("Mental Health", "Depression, SI,\npsychosis", COLORS[2], 3.4),
        ("Pediatric", "Any patient\nunder 18", COLORS[3], 2.0),
        ("General", "All other\npresentations", COLORS[4], 0.6),
    ]

    for name, desc, color, y_pos in routes:
        # Route box
        route_box = FancyBboxPatch((8, y_pos), 2.8, 1.0,
                                    boxstyle="round,pad=0.15", facecolor=color,
                                    edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85)
        ax.add_patch(route_box)
        ax.text(9.4, y_pos + 0.6, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(9.4, y_pos + 0.2, desc, ha="center", va="center",
                fontsize=7, color="#e0e0e0")

        # Arrow from classifier
        ax.annotate("", xy=(8, y_pos + 0.5), xytext=(6.2, 4.1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                    connectionstyle="arc3,rad=0.05"))

        # Output arrow
        output_box = FancyBboxPatch((11.5, y_pos + 0.1), 2.0, 0.8,
                                     boxstyle="round,pad=0.1", facecolor="#2a2a4e",
                                     edgecolor=GRID_COLOR, linewidth=1)
        ax.add_patch(output_box)
        ax.text(12.5, y_pos + 0.5, "Specialist\nAssessment", ha="center", va="center",
                fontsize=7, color="#888888")
        ax.annotate("", xy=(11.5, y_pos + 0.5), xytext=(10.8, y_pos + 0.5),
                    arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "routing_diagram.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'routing_diagram.png'}")


def fig_moderation_results():
    """Bar chart showing pass/block/flag results for different input types."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    input_types = ["Valid\nClinical Note", "Prompt\nInjection", "Non-Clinical\nInput",
                    "Abuse\nIndicators", "PII-Containing\nInput"]

    # Stacked bar data (each input type gets one outcome)
    pass_vals =  [1, 0, 0, 0, 0]
    flag_vals =  [0, 0, 0, 1, 0]
    block_vals = [0, 1, 1, 0, 1]

    x = np.arange(len(input_types))
    width = 0.5

    bars_pass = ax.bar(x, pass_vals, width, label="PASS", color=COLORS[1], alpha=0.85)
    bars_flag = ax.bar(x, flag_vals, width, bottom=pass_vals, label="FLAG", color=COLORS[3], alpha=0.85)
    bottom_for_block = [p + f for p, f in zip(pass_vals, flag_vals)]
    bars_block = ax.bar(x, block_vals, width, bottom=bottom_for_block, label="BLOCK", color=COLORS[4], alpha=0.85)

    # Labels on bars
    outcomes = ["PASS", "BLOCK", "BLOCK", "FLAG", "BLOCK"]
    outcome_colors = [COLORS[1], COLORS[4], COLORS[4], COLORS[3], COLORS[4]]
    for i, (outcome, color) in enumerate(zip(outcomes, outcome_colors)):
        ax.text(i, 0.5, outcome, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

    # Risk categories
    risk_cats = ["None", "Prompt\nInjection", "Non-Clinical\nContent", "Abuse\nReporting", "PII\nDetected"]
    for i, cat in enumerate(risk_cats):
        ax.text(i, -0.4, cat, ha="center", va="top",
                fontsize=7, color="#888888", style="italic")

    ax.set_ylabel("Moderation Decision", fontsize=12, color=TEXT_COLOR)
    ax.set_title("Input Moderation Results by Content Type",
                  fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(input_types, fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(-0.7, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["", ""], color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=10, facecolor="#2a2a4e",
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Summary text
    ax.text(2, 1.4, "3 Blocked  |  1 Flagged  |  1 Passed  |  100% Accuracy",
            ha="center", va="center", fontsize=10, color="#888888", style="italic")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "moderation_results.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'moderation_results.png'}")


if __name__ == "__main__":
    print("Generating figures for 02-building-systems...")
    fig_pipeline_architecture()
    fig_routing_diagram()
    fig_moderation_results()
    print("Done.")
