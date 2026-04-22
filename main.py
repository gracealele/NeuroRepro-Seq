
# =============================================================================
# MAIN PIPELINE
# =============================================================================
"""
======================================================================
HDLSS-Aware Transcriptomic Subtype Discovery for Postpartum Pychosis (PPP)
======================================================================

Design Goals
------------
1.  HDLSS-Aware       - Built for n << p (few sample, may genes).
                        Uses Ledoit-Wolf shrinkage, Sparse PCA, bootstrap
                        Stability filtering instead of naive PCA and K-means.

2.  Computationally   - Profiled runtime and RAM at every stage; a "low-resorce
    efficient           mode" caps memory and parallelism so the pipeline runs
                        on a standard laptop (<8GB RAM, no GPU)

3.  Small Sample      - Consensus clustering with bootstrap confidence intervals;
    Size                stability threshold rejects unreliable cluster solutions.
    
4.  Low-resource      - TMM-like normalization for sparse/low-coverage counts;
    sequencing          MAD-based variance filter robust to outlier samples.    
    
5.  PPP-Informed      - Literature-grounded gene sets (HPA axis, neuroinflamation
                        dopaminergic) upweighted in feature selection.

6.  Reproducible      - JSON config, structured logging, version-stamped outputs.

Usage
-----
    python ppp_pipeline.py                          # synthetic data
    python ppp_pipeline.py --config config.json     # custom config
    python ppp_pipeline.py --geo GSE152795          # real GEO data
    
Requirements (Standard)
    pip install pandas numpy scikit-learn scipy matplotlib seaborn
    
Requirements (Optional (recommended))
    pip install GEOparse umap-learn
    
GEO datasets relevant to PPP/ Perinatal psychosis:
    GSE152795   - postpatum mood/psychosis transcriptomics
    GSE116137   - bipolar disorder (first-episode) - closely related
    GSE181797   - peripheral blood in perinatal psychiatric disorders
"""


"""
ppp_subtypes/main.py
====================
Pipeline runner and CLI entry point for the PPP Subtype Pipeline.

Usage
-----
    # Run with synthetic data (default)
    python -m ppp_subtypes.main
    
    # Run with a custom config file
    python -m ppp_subtypes.main --config config.json
    
    # Run with real GEO data
    python -m ppp_subtypes.main --geo GSE152795
    
    # Combine: config + GEO override
    python -m ppp_subtypes.main --config config.json --geo GSE152795
"""
 
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Module imports ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
from modules.config             import PipelineConfig, setup_logging
from modules.profiler           import Profiler
from modules.data_loader        import load_data
from modules.preprocessing      import preprocess
from modules.dim_reduction      import hdlss_reduce, embed_2d
from modules.clustering         import (consensus_cluster, select_optimal_k, assign_subtypes)
from stability.characterisation import (marker_genes_mwu, geneset_enrichment, pathway_report)
from report.visualisation       import (plot_consensus_heatmaps, plot_embedding, plot_marker_heatmap, plot_k_selection, plot_geneset_scores, plot_compute_profile)
from report.reporter            import write_report



# =============================================================================
# PIPELINE RUNNER
# =============================================================================


def run(cfg: PipelineConfig) -> dict:
    """
    Execute the full PPP subtype discovery pipeline.

    Parameters
    ----------
    cfg : PipelineConfig

    Returns
    -------
    dict with keys:
        subtypes    : pd.Series – predicted subtype per sample
        markers     : pd.DataFrame – marker genes
        enrichment  : pd.DataFrame – gene-set enrichment scores
        profile     : pd.DataFrame – compute profile
        optimal_k   : int
        metrics     : dict – per-k cluster metrics
    """
    
    # Setup output directory FIRST
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)    
    
    # Setup logging
    setup_logging(out_dir)

    # Reproducibility
    np.random.seed(cfg.random_seed)
    
    # Profiler
    profiler = Profiler(
        active=cfg.profile_runtime or getattr(cfg, "profile_memory", False)
    )
 
    # Logging
    logging.info("=" * 68)
    logging.info("  PPP Molecular Subtype Pipeline  v2.0")
    logging.info(f" Output Dir   : {out_dir} ")
    logging.info(f" Random Seed  : {cfg.random_seed} ")
    logging.info(f" low Resource : {cfg.low_resource_mode} ")
    logging.info(f" n_iter       : {cfg.n_iterations}  ")
    logging.info(f" k_range      : {cfg.k_range} ")
    logging.info(f"  Profiling   : {profiler.active}")
    logging.info("=" * 68)
        
 
    # ── 1. Load ──────────────────────────────────────────────────────
    true_labels = None
    with profiler("1_data_loading"):
        if cfg.use_geo:
            expr_raw = load_geo(cfg.geo_id, cfg.geo_cache_dir)
        else:
            expr_raw, true_labels = generate_synthetic(cfg)
 
    # ── 2. Preprocess ────────────────────────────────────────────────
    with profiler("2_preprocessing"):
        expr = preprocess(expr_raw, cfg)
        if cfg.save_intermediates:
            expr.to_csv(out_dir / "preprocessed_expression.csv.gz", compression="gzip")
 
    # ── 3. Dim reduction ─────────────────────────────────────────────
    with profiler("3_dim_reduction"):
        coords   = hdlss_reduce(expr, cfg)
        embed_2d_ = embed_2d(coords, cfg)
 
    # ── 4. Consensus clustering ──────────────────────────────────────
    with profiler("4_consensus_clustering"):
        matrices = consensus_clustering(coords, cfg)
 
    # ── 5. Optimal k ─────────────────────────────────────────────────
    with profiler("5_k_selection"):
        optimal_k, metrics = select_optimal_k(matrices, coords, cfg)
 
    # ── 6. Assign subtypes ───────────────────────────────────────────
    with profiler("6_subtype_assignment"):
        subtypes = assign_subtypes(coords, optimal_k, expr.columns)
        logging.info(f"Subtype distribution:\n{subtypes.value_counts().to_string()}")
 
    # ── 7. Characterise ──────────────────────────────────────────────
    with profiler("7_characterisation"):
        markers     = marker_genes_mwu(expr, subtypes, top_n=25)
        enrichment  = geneset_enrichment_score(expr, subtypes)
        pathway_rep = pathway_report(subtypes, markers)
 
    # ── 8. Plot ──────────────────────────────────────────────────────
    with profiler("8_visualisation"):
        plot_consensus_heatmaps(matrices, optimal_k,
                                 out_dir / "consensus_heatmaps.png")
        plot_embedding(embed_2d_, subtypes, true_labels,
                       out_dir / "embedding.png")
        plot_marker_heatmap(expr, subtypes, markers,
                             out_dir / "marker_heatmap.png")
        plot_k_selection(metrics, optimal_k,
                          out_dir / "k_selection.png")
        plot_geneset_scores(enrichment,
                             out_dir / "geneset_enrichment.png")
 
    # ── 9. Save ──────────────────────────────────────────────────────
    profile_df = profiler.summary()
    with profiler("9_saving"):
        subtypes.to_csv(out_dir / "sample_subtypes.csv", header=True)
        markers.to_csv(out_dir / "marker_genes.csv", index=False)
        enrichment.to_csv(out_dir / "geneset_enrichment.csv")
        profile_df.to_csv(out_dir / "compute_profile.csv", index=False)
        plot_compute_profile(profile_df, out_dir / "compute_profile.png")
 
        if cfg.report:
            write_report(cfg, subtypes, metrics, optimal_k,
                         markers, pathway_rep, enrichment,
                         profile_df, out_dir / "report.txt")
 
    logging.info("\n✓  Pipeline complete.")
    logging.info(f"   Results → {out_dir.resolve()}/")
    total = profile_df["time_s"].sum()
    ram   = profile_df["peak_ram_mb"].max()
    logging.info(f"   Total runtime: {total:.1f}s | Peak RAM: {ram:.1f} MB\n")
    return subtypes, markers, enrichment
 
 
 
 
 
 
 
 
 
 
 
 
 
 


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run(cfg: PipelineConfig) -> dict:
    """
    Execute the full PPP subtype discovery pipeline.

    Parameters
    ----------
    cfg : PipelineConfig

    Returns
    -------
    dict with keys:
        subtypes    : pd.Series – predicted subtype per sample
        markers     : pd.DataFrame – marker genes
        enrichment  : pd.DataFrame – gene-set enrichment scores
        profile     : pd.DataFrame – compute profile
        optimal_k   : int
        metrics     : dict – per-k cluster metrics
    """
    
    setup_logging(cfg)
    np.random.seed(cfg.random_seed)
    out = cfg.out_path
    profiler = Profiler(active=cfg.profile_runtime)

    logging.info("=" * 68)
    logging.info("  PPP Molecular Subtype Pipeline  v2.0")
    logging.info(f"  low_resource={cfg.low_resource_mode}  "
                 f"n_iter={cfg.n_iterations}  "
                 f"k_range={cfg.k_range}")
    logging.info("=" * 68)

    # ── 1. Data loading ────────────────────────────────────────────────────────
    true_labels = None
    with profiler("1_data_loading"):
        expr_raw, true_labels = load_data(cfg)

     # ── 1. Load ──────────────────────────────────────────────────────
    true_labels = None
    with profiler("1_data_loading"):
        if cfg.use_geo:
            expr_raw = load_data(cfg.geo_id, cfg.geo_cache_dir)
        else:
            expr_raw, true_labels = generate_synthetic(cfg)
            
    # ── 2. Preprocessing ───────────────────────────────────────────────────────
    with profiler("2_preprocessing"):
        expr = preprocess(expr_raw, cfg)
        del expr_raw          # free memory immediately
        if cfg.save_intermediates:
            expr.to_csv(out / "preprocessed_expression.csv.gz",
                        compression="gzip")

    # ── 3. Dimensionality reduction ────────────────────────────────────────────
    with profiler("3_dim_reduction"):
        coords   = hdlss_reduce(expr, cfg)
        embed    = embed_2d(coords, cfg)

    # ── 4. Consensus clustering ────────────────────────────────────────────────
    with profiler("4_consensus_clustering"):
        matrices = consensus_cluster(coords, cfg)

    # ── 5. Optimal k selection ────────────────────────────────────────────────
    logging.info("\n[K] Computing stability metrics …")
    with profiler("5_k_selection"):
        optimal_k, metrics = select_optimal_k(matrices, coords, cfg)

    # ── 6. Subtype assignment ─────────────────────────────────────────────────
    with profiler("6_subtype_assignment"):
        subtypes = assign_subtypes(coords, optimal_k, expr.columns)
        logging.info("\n[Subtypes] Distribution:")
        logging.info(subtypes.value_counts().to_string())

    # ── 7. Biological characterisation ───────────────────────────────────────
    with profiler("7_characterisation"):
        markers    = marker_genes_mwu(expr, subtypes, cfg)
        enrichment = geneset_enrichment(expr, subtypes)
        pw_report  = pathway_report(subtypes, markers)

    # ── 8. Visualisation ─────────────────────────────────────────────────────
    with profiler("8_visualisation"):
        plot_consensus_heatmaps(
            matrices, optimal_k, out / "consensus_heatmaps.png", cfg)
        plot_embedding(
            embed, subtypes, true_labels, out / "embedding.png", cfg)
        plot_marker_heatmap(
            expr, subtypes, markers, out / "marker_heatmap.png", cfg)
        plot_k_selection(
            metrics, optimal_k, out / "k_selection.png", cfg)
        plot_geneset_scores(
            enrichment, out / "geneset_enrichment.png", cfg)

    # ── 9. Save results ───────────────────────────────────────────────────────
    profile_df = profiler.summary()
    with profiler("9_saving"):
        subtypes.to_csv(out / "sample_subtypes.csv", header=True)
        markers.to_csv(out / "marker_genes.csv", index=False)
        enrichment.to_csv(out / "geneset_enrichment.csv")
        profile_df.to_csv(out / "compute_profile.csv", index=False)
        cfg.to_json(str(out / "pipeline_config.json"))

        profiler.save(out, dpi=cfg.dpi)
        plot_compute_profile(profile_df, out / "compute_profile.png", cfg)

        if cfg.write_report:
            write_report(
                cfg, subtypes, metrics, optimal_k,
                markers, pw_report, enrichment,
                profile_df, out / "report.txt",
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    total_t = profile_df["time_s"].sum()
    peak_m  = profile_df["peak_ram_mb"].max()
    logging.info("\n" + "=" * 68)
    logging.info("  ✓  Pipeline complete")
    logging.info(f"     Optimal k    : {optimal_k}")
    logging.info(f"     Total runtime: {total_t:.1f}s")
    logging.info(f"     Peak RAM     : {peak_m:.0f} MB")
    logging.info(f"     Results      : {out.resolve()}/")
    logging.info("=" * 68 + "\n")

    return {
        "subtypes":   subtypes,
        "markers":    markers,
        "enrichment": enrichment,
        "profile":    profile_df,
        "optimal_k":  optimal_k,
        "metrics":    metrics,
    }


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ppp_subtypes",
        description="PPP Molecular Subtype Discovery Pipeline v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a JSON config file (optional)",
    )
    parser.add_argument(
        "--geo", default=None,
        help="GEO accession ID, e.g. GSE152795 (overrides config.use_geo)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (overrides config.out_dir)",
    )
    parser.add_argument(
        "--–n-samples", type=int, default=None,
        help="Number of synthetic samples (overrides config)",
    )
    parser.add_argument(
        "--k-max", type=int, default=None,
        help="Maximum k to test (overrides config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load config (file or defaults)
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()

    # CLI overrides
    if args.geo:
        cfg.use_geo = True
        cfg.geo_id  = args.geo

    if args.out:
        cfg.out_dir  = args.out
        cfg.out_path = Path(args.out)
        cfg.out_path.mkdir(exist_ok=True)

    if args.n_samples:
        cfg.synthetic_n_samples = args.n_samples

    if args.k_max:
        cfg.k_range = list(range(2, args.k_max + 1))

    run(cfg)
