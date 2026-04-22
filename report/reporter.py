"""
ppp_subtypes/modules/reporter.py
==================================
Generates a plain-text analysis report summarising the full pipeline run.

The report includes:
  • Pipeline configuration
  • Subtype discovery results (optimal k, distribution, per-k metrics)
  • Top marker genes per subtype (PPP genes starred)
  • PPP pathway annotations per subtype
  • Compute profile (runtime + RAM per stage)

Usage
-----
    from ppp_subtypes.modules.reporter import write_report
    write_report(cfg, subtypes, metrics, optimal_k,
                 markers, pathway_rep, enrichment,
                 profile_df, out_dir / "report.txt")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ppp_subtypes.modules.config import PipelineConfig


def write_report(
    cfg: PipelineConfig,
    subtypes: pd.Series,
    metrics: dict,
    optimal_k: int,
    markers: pd.DataFrame,
    pathway_rep: dict,
    enrichment: pd.DataFrame,
    profile_df: pd.DataFrame,
    out: Path,
) -> None:
    """
    Write a structured plain-text report of the pipeline run.

    Parameters
    ----------
    cfg         : PipelineConfig used for this run
    subtypes    : predicted subtype labels (sample → subtype)
    metrics     : per-k metrics dict from select_optimal_k
    optimal_k   : chosen k
    markers     : marker gene DataFrame from marker_genes_mwu
    pathway_rep : pathway annotation dict from pathway_report
    enrichment  : geneset enrichment pivot from geneset_enrichment
    profile_df  : profiler summary DataFrame
    out         : output file path
    """
    lines: list[str] = []
    sep = "=" * 68

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        sep,
        "  PPP Molecular Subtype Pipeline v2.0  –  Analysis Report",
        sep,
        "",
    ]

    # ── Configuration ─────────────────────────────────────────────────────────
    lines += [
        "CONFIGURATION",
        f"  Data mode          : "
        f"{'GEO  ' + cfg.geo_id if cfg.use_geo else 'Synthetic'}",
        f"  Samples (n)        : {subtypes.shape[0]}",
        f"  Normalisation      : {cfg.normalisation.upper()}",
        f"  Variance filter    : MAD (top {cfg.top_var_genes} genes)",
        f"  Dim. reduction     : {cfg.method} "
        f"({'Ledoit-Wolf' if cfg.ledoit_wolf_shrinkage else 'standard'})",
        f"  Components         : {cfg.n_components}",
        f"  Clustering         : Consensus k-means "
        f"({cfg.n_iterations} iters, sub-rate={cfg.subsample_rate:.0%})",
        f"  Bootstrap stability: {'Yes  (Jaccard ≥ ' + str(cfg.stability_threshold) + ')' if cfg.bootstrap_ci else 'Disabled'}",
        f"  PPP gene sets      : {'Yes  (weight=' + str(cfg.geneset_weighting) + ')' if cfg.use_ppp_genesets else 'No'}",
        f"  Low-resource mode  : {cfg.low_resource_mode}",
        "",
    ]

    # ── Subtype discovery ─────────────────────────────────────────────────────
    lines += ["SUBTYPE DISCOVERY", f"  Optimal k = {optimal_k}", ""]

    lines.append("  Per-k metrics:")
    for k in sorted(metrics):
        m   = metrics[k]
        row = (f"    k={k}  AUC={m['auc']:.4f}  "
               f"ΔAUC={m['delta_auc']:.4f}  "
               f"Sil={m['silhouette']:.3f}")
        if m["stability"] is not None:
            stable = "✓" if m["stability"] >= cfg.stability_threshold else "✗"
            row   += f"  Stab={m['stability']:.3f} {stable}"
        lines.append(row)

    lines += ["", "SUBTYPE DISTRIBUTION"]
    total = len(subtypes)
    for st, count in subtypes.value_counts().items():
        lines.append(f"  {st:<20} {count:>3} samples  ({100*count/total:.1f}%)")

    # ── Marker genes ──────────────────────────────────────────────────────────
    lines += ["", "TOP MARKER GENES  (MWU · rank-biserial effect size)"]
    lines.append("  ★ = PPP literature gene")
    for st in sorted(subtypes.unique()):
        st_df = (markers[markers["subtype"] == st]
                 .nlargest(10, "effect_size")
                 [["gene", "effect_size", "log2fc", "is_ppp_gene"]])
        lines.append(f"\n  {st}:")
        for _, row in st_df.iterrows():
            tag  = " ★" if row["is_ppp_gene"] else "  "
            fc   = f"{row['log2fc']:+.2f}"
            lines.append(
                f"    {row['gene']:<12}{tag}  ES={row['effect_size']:.3f}  "
                f"log2FC={fc}"
            )

    # ── PPP pathway annotations ───────────────────────────────────────────────
    lines += ["", "PPP PATHWAY ANNOTATIONS"]
    for st, hits in pathway_rep.items():
        lines.append(f"\n  {st}:")
        if hits:
            for gs_name, genes, pathway_hints in hits:
                lines.append(f"    [{gs_name}]  overlapping genes: "
                              f"{', '.join(genes[:6])}")
                for hint in pathway_hints[:3]:
                    lines.append(f"      → {hint}")
        else:
            lines.append("    No PPP gene set overlap detected")

    # ── Gene set enrichment ───────────────────────────────────────────────────
    lines += ["", "GENE SET ENRICHMENT SCORES  (mean z-score per subtype)"]
    if not enrichment.empty:
        lines.append("  " + "  ".join(f"{c:>14}" for c in enrichment.columns))
        for gs, row in enrichment.iterrows():
            vals = "  ".join(f"{v:>14.3f}" if pd.notna(v) else f"{'NA':>14}"
                             for v in row)
            lines.append(f"  {gs:<28}  {vals}")

    # ── Compute profile ───────────────────────────────────────────────────────
    lines += ["", "COMPUTE PROFILE"]
    if not profile_df.empty:
        for _, row in profile_df.iterrows():
            lines.append(
                f"  {row['stage']:<38}  {row['time_s']:>7.2f}s  "
                f"{row['peak_ram_mb']:>7.1f} MB"
            )
        total_t = profile_df["time_s"].sum()
        peak_m  = profile_df["peak_ram_mb"].max()
        lines += [
            f"\n  Total runtime  : {total_t:.1f}s",
            f"  Peak RAM usage : {peak_m:.0f} MB",
        ]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "",
        "NOTE: Designed for low-resource clinical environments.",
        "      PPP gene signatures sourced from published literature.",
        "      This is a discovery pipeline; results require clinical validation.",
        sep,
    ]

    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    logging.info(f"[Report] Written → {out}")