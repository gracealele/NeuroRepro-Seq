# ppp_subtypes/modules/__init__.py
"""
ppp_subtypes.modules
====================
Internal modules of the PPP Subtype Pipeline.

Each module has a single responsibility and can be imported independently
for use in custom workflows or notebooks.

    from ppp_subtypes.modules.preprocessing import preprocess
    from ppp_subtypes.modules.clustering    import consensus_cluster

Dependency order (each module only imports from those above it):
    config -> genesets -> data_loader -> preprocessing
           -> dim_reduction -> clustering -> characterisation
           -> visualisation -> reporter -> main
"""

from ppp_subtypes.modules.config           import PipelineConfig
from ppp_subtypes.modules.genesets         import PPP_GENESETS, get_all_ppp_genes
from ppp_subtypes.modules.data_loader      import load_data, generate_synthetic
from ppp_subtypes.modules.preprocessing    import preprocess
from ppp_subtypes.modules.dim_reduction    import hdlss_reduce, embed_2d
from ppp_subtypes.modules.clustering       import consensus_cluster, select_optimal_k, assign_subtypes
from ppp_subtypes.modules.characterisation import marker_genes_mwu, geneset_enrichment, pathway_report
from ppp_subtypes.modules.visualisation    import (plot_consensus_heatmaps, plot_embedding,
                                                   plot_marker_heatmap, plot_k_selection,
                                                   plot_geneset_scores, plot_compute_profile)
from ppp_subtypes.modules.reporter         import write_report
from ppp_subtypes.modules.profiler         import Profiler

__all__ = [
    "PipelineConfig",
    "PPP_GENESETS", "get_all_ppp_genes",
    "load_data", "generate_synthetic",
    "preprocess",
    "hdlss_reduce", "embed_2d",
    "consensus_cluster", "select_optimal_k", "assign_subtypes",
    "marker_genes_mwu", "geneset_enrichment", "pathway_report",
    "plot_consensus_heatmaps", "plot_embedding", "plot_marker_heatmap",
    "plot_k_selection", "plot_geneset_scores", "plot_compute_profile",
    "write_report",
    "Profiler",
]
