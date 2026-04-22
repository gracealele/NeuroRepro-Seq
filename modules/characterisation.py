from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from modules.config import PipelineConfig
from modules.genesets import (
    PPP_GENESETS, PATHWAY_HINTS, get_all_ppp_genes
)


# MARKER GENE IDENTIFICATION

def marker_genes_mwu(
    expr: pd.DataFrame,
    subtypes: pd.Series,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    
    """
    One-vs-rest Mann-Whitney U test for every gene × subtype combination.

    For each (gene, subtype) pair:
      - in-group:  samples in this subtype
      - out-group: all other samples

    Returns the top cfg.marker_top_n genes per subtype, sorted by
    effect size descending, with PPP genes prioritised at ties.

    Output columns:
    subtype, gene, mwu_stat, p_value, effect_size (RBC),
    mean_in, mean_out, is_ppp_gene
    """
    
    logging.info(
        "[Char] Mann-Whitney U marker gene identification "
        "(non-parametric, one-vs-rest) …"
    )
    ppp_set      = set(get_all_ppp_genes())
    marker_rows: list[dict] = []

    for st in sorted(subtypes.unique()):
        in_mask  = subtypes == st
        out_mask = ~in_mask

        # Guard: need at least 2 samples per group for MWU
        if in_mask.sum() < 2 or out_mask.sum() < 2:
            logging.warning(
                f"[Char] {st}: too few samples for MWU "
                f"(in={in_mask.sum()}, out={out_mask.sum()}) – skipping"
            )
            continue

        in_expr  = expr.loc[:, in_mask]
        out_expr = expr.loc[:, out_mask]

        for gene in expr.index:
            x_in  = in_expr.loc[gene].values
            x_out = out_expr.loc[gene].values

            # Skip genes with zero variance in both groups (uninformative)
            if x_in.std() < 1e-6 and x_out.std() < 1e-6:
                continue

            stat, pval = mannwhitneyu(x_in, x_out, alternative="greater")

            # Rank-biserial correlation as effect size
            # RBC = 1 − (2 × U) / (n1 × n2)
            # Range: −1 (always lower) to +1 (always higher)
            n1, n2     = len(x_in), len(x_out)
            effect_rbc = 1.0 - (2.0 * stat) / (n1 * n2)

            marker_rows.append({
                "subtype":     st,
                "gene":        gene,
                "mwu_stat":    float(stat),
                "p_value":     float(pval),
                "effect_size": float(effect_rbc),
                "mean_in":     float(x_in.mean()),
                "mean_out":    float(x_out.mean()),
                "log2fc":      float(x_in.mean() - x_out.mean()),
                "is_ppp_gene": gene in ppp_set,
            })

    df = pd.DataFrame(marker_rows)

    # Sort: subtype → PPP gene priority → effect size descending
    df = df.sort_values(
        ["subtype", "is_ppp_gene", "effect_size"],
        ascending=[True, False, False],
    )
    top = df.groupby("subtype").head(cfg.marker_top_n).reset_index(drop=True)

    # Log summary
    for st in sorted(top["subtype"].unique()):
        st_df   = top[top["subtype"] == st]
        ppp_cnt = st_df["is_ppp_gene"].sum()
        top5    = st_df["gene"].head(5).tolist()
        logging.info(
            f"  {st}: top genes = {top5}  ({ppp_cnt} PPP genes in top {cfg.marker_top_n})"
        )

    return top


# GENE SET ENRICHMENT SCORING

def geneset_enrichment(
    expr: pd.DataFrame,
    subtypes: pd.Series,
) -> pd.DataFrame:
    """
    Single-sample gene-set enrichment scoring for PPP signatures.

    Method: mean z-score of signature genes per subtype (ssGSEA-lite).
    Robust and interpretable without requiring permutation testing.

    Returns:
    pivot DataFrame: gene_sets × subtypes  (relative enrichment scores)
    """
    logging.info("[Char] Computing PPP gene-set enrichment scores …")

    # Z-score each gene across all samples
    z_expr = expr.apply(lambda row: (row - row.mean()) / (row.std() + 1e-9), axis=1)

    rows: list[dict] = []
    for st in sorted(subtypes.unique()):
        sub_expr  = expr.loc[:, subtypes == st]
        sub_z     = z_expr.loc[:, subtypes == st]

        for gs_name, gs_genes in PPP_GENESETS.items():
            present = [g for g in gs_genes if g in z_expr.index]
            if present:
                score = float(sub_z.loc[present].mean().mean())
            else:
                score = np.nan
            rows.append({
                "subtype":         st,
                "geneset":         gs_name,
                "enrichment_score": score,
                "n_genes_matched": len(present),
            })

    df    = pd.DataFrame(rows)
    pivot = df.pivot(index="geneset", columns="subtype",
                     values="enrichment_score")

    logging.info(
        f"[Char] Gene set enrichment: {pivot.shape[0]} sets × "
        f"{pivot.shape[1]} subtypes"
    )
    return pivot


# PATHWAY REPORT

def pathway_report(
    subtypes: pd.Series,
    markers: pd.DataFrame,
) -> dict[str, list[tuple]]:
    """
    Map each subtype's top PPP marker genes to biological pathway annotations.
    For each subtype:
      - Extract marker genes that are PPP signature genes.
      - Check which PPP gene sets they belong to.
      - Return the matching sets and their PATHWAY_HINTS descriptions.

    Returns:
    dict: {subtype_name: [(geneset_name, [genes], [pathway_hints])]}
    """
    report: dict[str, list[tuple]] = {}

    for st in sorted(subtypes.unique()):
        st_markers = markers[markers["subtype"] == st]
        ppp_hits   = st_markers[st_markers["is_ppp_gene"]]["gene"].tolist()

        matched: list[tuple] = []
        for gs_name, gs_genes in PPP_GENESETS.items():
            overlap = [g for g in ppp_hits if g in gs_genes]
            if overlap:
                hints = PATHWAY_HINTS.get(gs_name, [])
                matched.append((gs_name, overlap, hints))

        report[st] = matched

        if matched:
            logging.info(f"  {st}: {len(matched)} PPP pathways hit")
            for gs_name, genes, _ in matched:
                logging.info(f"    [{gs_name}]  {', '.join(genes[:4])}")
        else:
            logging.info(f"  {st}: no PPP signature gene overlap in top markers")

    return report