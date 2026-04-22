# ppp_subtypes/__init__.py
"""
Quick start
-----------
    from ppp_subtypes import run, PipelineConfig

    cfg     = PipelineConfig(synthetic_n_samples=40, k_range=[2, 3, 4])
    results = run(cfg)

Modules
-------
    main                     - pipeline runner + CLI
    modules.config           - PipelineConfig dataclass
    modules.profiler         - runtime + memory profiler
    modules.genesets         - PPP literature gene signatures
    modules.data_loader      - GEO download + synthetic data
    modules.preprocessing   - TMM norm, MAD filter
    modules.dim_reduction   - Ledoit-Wolf + PCA
    modules.clustering      - consensus clustering + stability
    modules.characterisation - MWU markers + enrichment
    modules.visualisation   - all plot functions
    modules.reporter        - plain-text report writer
"""

__version__ = "2.0.0"
__author__  = "Grace Alele"
__license__ = "MIT"

from main import run
from modules.config import PipelineConfig
from modules.genesets import PPP_GENESETS, get_all_ppp_genes, geneset_overlap

__all__ = ["run", "PipelineConfig", "PPP_GENESETS", "get_all_ppp_genes", "geneset_overlap"]
