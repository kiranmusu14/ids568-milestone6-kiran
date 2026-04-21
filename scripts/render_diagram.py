#!/usr/bin/env python3
"""
scripts/render_diagram.py
Renders the RAG pipeline architecture as a PNG using matplotlib.
Output: rag_pipeline_diagram.png (project root)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Layout constants ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 14, 10
OUT_PATH = Path(__file__).parent.parent / "rag_pipeline_diagram.png"

# Colour palette
C_OFFLINE  = "#D6EAF8"   # light blue  — indexing phase boxes
C_ONLINE   = "#D5F5E3"   # light green — query phase boxes
C_DECISION = "#FDEBD0"   # light orange — decision point
C_EDGE     = "#2C3E50"   # dark grey   — box borders & arrows
C_LABEL    = "#2C3E50"
C_PHASE    = "#85929E"   # phase label text


def box(ax, x, y, w, h, label, sublabel=None, color=C_OFFLINE, fontsize=9):
    """Draw a rounded box with an optional sublabel."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        linewidth=1.4,
        edgecolor=C_EDGE,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w / 2, cy + 0.09, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=C_LABEL, zorder=4)
        ax.text(x + w / 2, cy - 0.13, sublabel,
                ha="center", va="center", fontsize=7.5,
                color="#555555", zorder=4, style="italic")
    else:
        ax.text(x + w / 2, cy, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=C_LABEL, zorder=4)


def arrow(ax, x0, y0, x1, y1, label=None):
    """Draw a straight arrow between two points."""
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_EDGE,
            lw=1.5,
            mutation_scale=14,
        ),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.04, my, label,
                ha="left", va="center", fontsize=7.5,
                color="#666666", zorder=5)


def phase_label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=8.5, color=C_PHASE,
            fontweight="bold",
            fontstyle="italic")


# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, 9.65, "RAG Pipeline — Architecture",
        ha="center", va="center", fontsize=14,
        fontweight="bold", color=C_LABEL)

# ═══════════════════════════════════════════════════════════════════════════════
# INDEXING PHASE (top half, y = 6.8 – 9.2)
# ═══════════════════════════════════════════════════════════════════════════════
phase_label(ax, 0.2, 9.3, "INDEXING PHASE  (offline — run once)")
ax.axhline(y=9.2, xmin=0.01, xmax=0.99, color=C_PHASE, lw=0.8, ls="--")

BW, BH = 2.8, 0.8   # box width / height
BY = 7.8             # box y

# Documents
box(ax, 0.5, BY, BW, BH, "Raw Documents",
    sublabel="10 MLOps/AI docs (in-memory str)",
    color=C_OFFLINE)

# Chunker
box(ax, 4.0, BY, BW, BH, "Chunker",
    sublabel="Fixed-size · 512 chars · 100 overlap",
    color=C_OFFLINE)

# Embedder
box(ax, 7.5, BY, BW, BH, "Embedder",
    sublabel="all-MiniLM-L6-v2 · 384-dim · L2-norm",
    color=C_OFFLINE)

# FAISS
box(ax, 11.0, BY, BW, BH, "FAISS IndexFlatIP",
    sublabel="33 vectors · exact inner-product · RAM",
    color=C_OFFLINE)

# Arrows (indexing)
arrow(ax, 0.5 + BW, BY + BH / 2, 4.0, BY + BH / 2,
      label="raw text")
arrow(ax, 4.0 + BW, BY + BH / 2, 7.5, BY + BH / 2,
      label="chunks[]")
arrow(ax, 7.5 + BW, BY + BH / 2, 11.0, BY + BH / 2,
      label="embeddings\n[33×384]")

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY PHASE (bottom half, y = 0.4 – 6.5)
# ═══════════════════════════════════════════════════════════════════════════════
phase_label(ax, 0.2, 7.1, "QUERY PHASE  (online — per user request)")
ax.axhline(y=7.0, xmin=0.01, xmax=0.99, color=C_PHASE, lw=0.8, ls="--")

# Column x-centres and widths for query phase
QBW, QBH = 3.0, 0.75

# Vertical stack (left-to-right as top-to-bottom pipeline)
nodes = [
    # (x_left, y_bottom, label, sublabel, color)
    (0.3,  5.6, "User Query",        None,                           C_ONLINE),
    (0.3,  4.5, "Query Embedder",    "all-MiniLM-L6-v2 · ~0.2–0.8 s incl. encoding",  C_ONLINE),
    (0.3,  3.4, "Retriever",         "FAISS.search(q, k=3) · exact IP search", C_ONLINE),
    (0.3,  2.3, "Context Assembly",  "Concat chunks · format prompt", C_DECISION),
    (0.3,  1.2, "Generator",         "mistral:7b-instruct via Ollama · ~5–17 s", C_ONLINE),
    (0.3,  0.1, "Grounded Answer",   None,                           C_ONLINE),
]

for (xb, yb, lbl, sub, col) in nodes:
    box(ax, xb, yb, QBW, QBH, lbl, sublabel=sub, color=col)

# Vertical arrows (query phase)
for i in range(len(nodes) - 1):
    x_mid = nodes[i][0] + QBW / 2
    y_top = nodes[i][1]          # bottom of upper box → top
    y_bot = nodes[i + 1][1] + QBH  # top of lower box
    arrow(ax, x_mid, y_top, x_mid, y_bot)

# FAISS index feeds into Retriever (cross-phase arrow)
# FAISS box right edge at 11.0+2.8=13.8 — drop down to Retriever
faiss_cx = 11.0 + BW / 2
retriever_right_x = 0.3 + QBW
retriever_cy = 3.4 + QBH / 2

ax.annotate(
    "",
    xy=(retriever_right_x, retriever_cy),
    xytext=(faiss_cx, BY),   # from bottom of FAISS box
    arrowprops=dict(
        arrowstyle="-|>",
        color="#8E44AD",
        lw=1.5,
        mutation_scale=13,
        connectionstyle="arc3,rad=0.25",
    ),
    zorder=2,
)
ax.text(9.5, 5.3, "index lookup\n(at query time)",
        ha="center", va="center", fontsize=7.5,
        color="#8E44AD", style="italic")

# ── Decision branch annotation (Context Assembly) ─────────────────────────────
dec_x = 0.3 + QBW + 0.15   # just right of Context Assembly box
dec_y = 2.3 + QBH / 2

ax.annotate(
    "Context sufficient?\n  Yes → Generator\n  No  → expand k / low-conf flag",
    xy=(dec_x, dec_y),
    xytext=(dec_x + 1.8, dec_y),
    fontsize=7.5,
    color="#C0392B",
    va="center",
    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#C0392B", lw=0.8),
    zorder=5,
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=C_OFFLINE,  edgecolor=C_EDGE, label="Indexing component"),
    mpatches.Patch(facecolor=C_ONLINE,   edgecolor=C_EDGE, label="Query component"),
    mpatches.Patch(facecolor=C_DECISION, edgecolor=C_EDGE, label="Decision point"),
]
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=8, framealpha=0.9, edgecolor=C_EDGE)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
