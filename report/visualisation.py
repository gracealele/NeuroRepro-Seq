"""
ppp_subtypes/modules/visualisation.py
========================================
All visualisation functions for the PPP Subtype Pipeline.

Six plots are produced:
    plot_consensus_heatmaps  – consensus matrices for each k tested
    plot_embedding           – 2-D scatter coloured by subtype (+ true labels)
    plot_marker_heatmap      – expression heatmap of top marker genes
    plot_k_selection         – AUC, silhouette, stability vs k
    plot_geneset_scores      – PPP gene-set enrichment heatmap per subtype
    plot_compute_profile     – runtime and RAM bar charts from profiler

All functions:
  • write PNG to a provided Path
  • close the figure immediately to free RAM
  • are safe to call individually or via the pipeline runner

Usage
-----
    from ppp_subtypes.modules.visualisation import (
        plot_consensus_heatmaps, plot_embedding, ...
    )
    plot_consensus_heatmaps(matrices, optimal_k, out_dir / "consensus.png", cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ppp_subtypes.modules.config import PipelineConfig
from ppp_subtypes.modules.genesets import get_all_ppp_genes

# ── Design constants ─────────────────────────────────────────────────────────
PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#9B5DE5"]


def _subtype_colors(subtypes: pd.Series) -> dict[str, str]:
    return {
        s: PALETTE[i % len(PALETTE)]
        for i, s in enumerate(sorted(subtypes.unique()))
    }


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[Plot] Saved → {path}")


# =============================================================================
# CONSENSUS HEATMAPS
# =============================================================================

def plot_consensus_heatmaps(
    matrices: dict[int, np.ndarray],
    optimal_k: int,
    out: Path,
    cfg: PipelineConfig,
) -> None:
    """
    One heatmap panel per k.  Optimal k is highlighted with a red border.
    A bimodal (0/1) matrix indicates clean, well-separated clusters.
    """
    ks  = sorted(matrices)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.5 * len(ks), 4.5))
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        mat = matrices[k]
        sns.heatmap(
            mat, ax=ax, cmap="Blues", vmin=0, vmax=1,
            xticklabels=False, yticklabels=False,
            cbar_kws={"shrink": 0.6, "label": "Co-cluster probability"},
        )
        is_opt = k == optimal_k
        for sp in ax.spines.values():
            sp.set_edgecolor("#E63946" if is_opt else "none")
            sp.set_linewidth(3 if is_opt else 0)
        ax.set_title(
            f"k={k}" + (" ← optimal" if is_opt else ""),
            fontweight="bold" if is_opt else "normal",
            fontsize=11,
        )

    fig.suptitle(
        "Consensus Matrices – PPP Molecular Subtype Discovery",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out, cfg.dpi)


# =============================================================================
# 2-D EMBEDDING
# =============================================================================

def plot_embedding(
    embed: np.ndarray,
    subtypes: pd.Series,
    true_labels: Optional[pd.Series],
    out: Path,
    cfg: PipelineConfig,
) -> None:
    """
    Scatter plot of the 2-D embedding coloured by predicted subtypes.
    If true_labels is provided (synthetic data), a second panel shows them.
    """
    ncols = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    cmap = _subtype_colors(subtypes)

    def _panel(ax: plt.Axes, labels: list[str], title: str, color_map: dict) -> None:
        for st, col in color_map.items():
            mask = np.array([l == st for l in labels])
            ax.scatter(
                embed[mask, 0], embed[mask, 1],
                c=col, label=st, s=70, alpha=0.85,
                edgecolors="white", linewidths=0.5,
            )
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Dim-1"); ax.set_ylabel("Dim-2")
        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_facecolor("#F5F5F5")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    _panel(axes[0], subtypes.tolist(), "Predicted Subtypes", cmap)

    if true_labels is not None:
        true_cmap = _subtype_colors(true_labels)
        _panel(
            axes[1], true_labels.tolist(),
            "True Subtypes (synthetic reference)", true_cmap,
        )

    fig.suptitle(
        "Embedding – PPP Molecular Subtypes",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out, cfg.dpi)


# =============================================================================
# MARKER GENE HEATMAP
# =============================================================================

def plot_marker_heatmap(
    expr: pd.DataFrame,
    subtypes: pd.Series,
    markers: pd.DataFrame,
    out: Path,
    cfg: PipelineConfig,
    top_n: int = 15,
) -> None:
    """
    Expression heatmap of top marker genes per subtype.
    PPP signature genes are highlighted in red on the y-axis.
    Samples are ordered by predicted subtype.
    """
    # Collect top genes per subtype
    top_genes: list[str] = []
    for st in sorted(subtypes.unique()):
        st_genes = (
            markers[markers["subtype"] == st]
            .nlargest(top_n, "effect_size")["gene"]
            .tolist()
        )
        top_genes.extend(st_genes)
    top_genes = list(dict.fromkeys(g for g in top_genes if g in expr.index))

    if not top_genes:
        logging.warning("[Plot] No marker genes to plot – skipping heatmap")
        return

    col_order = subtypes.sort_values().index
    sub_expr  = expr.loc[top_genes, col_order]

    cmap_dict  = _subtype_colors(subtypes)
    ppp_set    = set(get_all_ppp_genes())

    fig, ax = plt.subplots(
        figsize=(
            max(10, sub_expr.shape[1] * 0.20 + 4),
            max(8,  len(top_genes)    * 0.30 + 2),
        )
    )
    sns.heatmap(
        sub_expr, ax=ax, cmap="RdBu_r", center=0,
        xticklabels=False, yticklabels=True,
        cbar_kws={"label": "Normalised expression", "shrink": 0.6},
    )

    # Highlight PPP genes in red
    for label in ax.get_yticklabels():
        if label.get_text() in ppp_set:
            label.set_color("#E63946")
            label.set_fontweight("bold")

    handles = [
        mpatches.Patch(color=col, label=st)
        for st, col in cmap_dict.items()
    ]
    ax.legend(
        handles=handles, title="Subtype",
        loc="upper right", bbox_to_anchor=(1.18, 1), fontsize=8,
    )
    ax.set_title(
        "Marker Genes per PPP Subtype\n"
        "(MWU ranked · red = PPP signature gene)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel(f"Samples (n={sub_expr.shape[1]}, ordered by subtype)")
    plt.tight_layout()
    _save(fig, out, cfg.dpi)


# =============================================================================
# K-SELECTION MULTI-CRITERION PLOT
# =============================================================================

def plot_k_selection(
    metrics: dict,
    optimal_k: int,
    out: Path,
    cfg: PipelineConfig,
) -> None:
    """
    Three-panel plot: Consensus AUC, Silhouette score, Bootstrap stability.
    Vertical dashed line marks the chosen optimal k.
    """
    ks   = sorted(metrics)
    aucs = [metrics[k]["auc"]        for k in ks]
    sils = [metrics[k]["silhouette"] for k in ks]
    stbs = [metrics[k]["stability"]  for k in ks]

    has_stability = all(s is not None for s in stbs)
    nrows = 3 if has_stability else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(6, 3.2 * nrows), sharex=True)

    def _line(ax: plt.Axes, y: list, label: str, colour: str,
              ylabel: str, hline: Optional[float] = None) -> None:
        ax.plot(ks, y, "o-", color=colour, linewidth=2.2, markersize=8, label=label)
        ax.axvline(optimal_k, color="#E63946", linestyle="--",
                   linewidth=1.5, label=f"Optimal k={optimal_k}")
        if hline is not None:
            ax.axhline(hline, color="grey", linestyle=":", linewidth=1.2,
                       alpha=0.7, label=f"Threshold={hline}")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_facecolor("#F8F8F8")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    _line(axes[0], aucs, "Consensus CDF AUC", "#457B9D", "AUC")
    _line(axes[1], sils, "Silhouette",         "#2A9D8F", "Silhouette Score", 0)
    if has_stability:
        _line(axes[2], stbs, "Bootstrap Jaccard",
              "#9B5DE5", "Stability (Jaccard)", cfg.stability_threshold)

    axes[-1].set_xlabel("Number of clusters (k)", fontsize=10)
    axes[-1].set_xticks(ks)
    fig.suptitle(
        "Optimal k Selection – Multi-criterion",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out, cfg.dpi)


# =============================================================================
# GENE SET ENRICHMENT HEATMAP
# =============================================================================

def plot_geneset_scores(
    enrichment: pd.DataFrame,
    out: Path,
    cfg: PipelineConfig,
) -> None:
    """
    Heatmap of PPP gene-set enrichment scores per subtype.
    Values are normalised row-wise so that the dominant subtype per
    gene set is clearly visible.
    """
    normed = enrichment.div(enrichment.abs().max(axis=1) + 1e-9, axis=0)

    fig, ax = plt.subplots(figsize=(max(6, normed.shape[1] * 1.5 + 2), 4.5))
    sns.heatmap(
        normed, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
        vmin=0, vmax=1,
        cbar_kws={"label": "Relative enrichment (row-normalised)"},
    )
    ax.set_title(
        "PPP Gene-Set Enrichment per Subtype",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Predicted Subtype")
    ax.set_ylabel("PPP Gene Set")
    plt.tight_layout()
    _save(fig, out, cfg.dpi)


# =============================================================================
# COMPUTE PROFILE
# =============================================================================

def plot_compute_profile(
    profile_df: pd.DataFrame,
    out: Path,
    cfg: PipelineConfig,
) -> None:
    """
    Two-panel bar chart: runtime (s) and peak RAM (MB) per pipeline stage.
    Useful for demonstrating computational efficiency in low-resource settings.
    """
    if profile_df.empty:
        return

    colours = [f"C{i}" for i in range(len(profile_df))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(3, len(profile_df) * 0.45)))

    ax1.barh(profile_df["stage"], profile_df["time_s"], color=colours)
    ax1.set_xlabel("Wall-clock time (s)")
    ax1.set_title("Runtime per Stage", fontweight="bold")
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    ax2.barh(profile_df["stage"], profile_df["peak_ram_mb"], color=colours)
    ax2.set_xlabel("Peak RAM (MB)")
    ax2.set_title("Memory per Stage", fontweight="bold")
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    total_t = profile_df["time_s"].sum()
    peak_m  = profile_df["peak_ram_mb"].max()
    fig.suptitle(
        f"Compute Profile  ·  Total {total_t:.1f}s  |  Peak {peak_m:.0f} MB",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out, cfg.dpi)