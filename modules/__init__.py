# ppp_subtypes/modules/__init__.py
from .config           import PipelineConfig
from .genesets         import PPP_GENESETS, get_all_ppp_genes
from .data_loader      import load_data, generate_synthetic
from .preprocessing    import preprocess
from .dim_reduction    import hdlss_reduce, embed_2d
from .clustering       import consensus_cluster, select_optimal_k, assign_subtypes
from .characterisation import marker_genes_mwu, geneset_enrichment, pathway_report
from .visualisation    import (plot_consensus_heatmaps, plot_embedding,
                               plot_marker_heatmap, plot_k_selection,
                               plot_geneset_scores, plot_compute_profile)
from .reporter         import write_report
from .profiler         import Profiler