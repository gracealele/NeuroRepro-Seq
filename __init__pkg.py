# ppp_subtypes/__init__.py
"""
ppp_subtypes
============
Molecular Disease Subtype Discovery Pipeline for Postpartum Psychosis.

Designed for:
  * Small sample sizes (n = 10-80, HDLSS regime)
  * Low-resource clinical environments (<=1.5 GB RAM, CPU-only)
  * PPP-specific biology (literature-curated gene signatures)

Quick start
-----------
    from ppp_subtypes import run, PipelineConfig

    cfg     = PipelineConfig(synthetic_n_samples=40, k_range=[2, 3, 4])
    results = run(cfg)

Modules
-------
    ppp_subtypes.main                     - pipeline runner + CLI
    ppp_subtypes.modules.config           - PipelineConfig dataclass
    ppp_subtypes.modules.profiler         - runtime + memory profiler
    ppp_subtypes.modules.genesets         - PPP literature gene signatures
    ppp_subtypes.modules.data_loader      - GEO download + synthetic data
    ppp_subtypes.modules.preprocessing   - TMM norm, MAD filter
    ppp_subtypes.modules.dim_reduction   - Ledoit-Wolf + PCA
    ppp_subtypes.modules.clustering      - consensus clustering + stability
    ppp_subtypes.modules.characterisation - MWU markers + enrichment
    ppp_subtypes.modules.visualisation   - all plot functions
    ppp_subtypes.modules.reporter        - plain-text report writer
"""

__version__ = "2.0.0"
__author__  = "Grace Alele"
__license__ = "MIT"

from ppp_subtypes.main import run
from ppp_subtypes.modules.config import PipelineConfig
from ppp_subtypes.modules.genesets import PPP_GENESETS, get_all_ppp_genes, geneset_overlap

__all__ = ["run", "PipelineConfig", "PPP_GENESETS", "get_all_ppp_genes", "geneset_overlap"]
