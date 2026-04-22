"""
ppp_subtypes/modules/data_loader.py
=====================================
Data loading for the PPP pipeline.
 
Two modes
---------
1. GEO download  – downloads a real GEO series via GEOparse,
                   caches to disk, returns (genes * samples) DataFrame.
 
2. Synthetic     – generates a biologically realistic count matrix
                   anchored to the PPP gene signatures, in the HDLSS
                   regime (few samples, many genes) to stress-test the
                   downstream pipeline.
 
Usage
-----
    from ppp_subtypes.modules.data_loader import load_data
    from ppp_subtypes.modules.config import PipelineConfig
 
    cfg  = PipelineConfig()
    expr, true_labels = load_data(cfg)
    # true_labels is None when use_geo=True
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
 
from modules.config import PipelineConfig
from modules.genesets import PPP_GENESETS, get_all_ppp_genes


# GEO DOWNLOAD


def load_geo(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Download a GEO series and return a (genes * samples) expression DataFrame.
 
    Results are pickled to cfg.geo_cache_dir so subsequent runs are instant.
 
    Requires:  pip install GEOparse
 
    Recommended accessions for PPP:
        GSE152795  – postpartum mood / psychosis transcriptomics
        GSE116137  – bipolar disorder first-episode (related)
        GSE181797  – peripheral blood in perinatal psychiatric disorders
        GSE54913   – postpartum depression transcriptomics
    """    
    
    try:
        import GEOparse
    except ImportError as exc:
        raise ImportError(
            "GEOparse is required for GEO download.  "
            "Install with:  pip install GEOparse"
        ) from exc

    cache_dir = Path(cfg.geo_cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_pkl = cache_dir / f"{cfg.geo_id}.pkl"
 
    if cache_pkl.exists():
        logging.info(f"[Data] Loading cached {cfg.geo_id} from {cache_pkl} …")
        return pd.read_pickle(cache_pkl)
    

    logging.info(f"[Data] Downloading {cfg.geo_id} from GEO ...")
    gse = GEOparse.get_GEO(geo=cfg.geo_id, destdir=str(cache_dir), silent=True)
    
    frames = []
    for gsm_name, gsm in gse.gsms.items():
        if gsm.table is not None and not gsm.table.empty:
            col = gsm.table.set_index(gsm.table.columns[0])["VALUE"]
            col.name = gsm_name
            frames.append(col)
    
    if not frames:
        raise ValueError(f"No expression tables found in {cfg.geo_id}")


    # Extract expression data (assuming it's in the first GSM)
    expr = pd.concat(frames, axis=1).dropna()
    expr.index.name = "gene_id"
    expr.to_pickle(cache_pkl)
    logging.info(
        f"[Data] GEO loaded: {expr.shape[0]} probes * {expr.shape[1]} samples"
        f" (cached → {cache_pkl})"
    )
    return expr


# =============================================================================
# SYNTHETIC DATA
# =============================================================================
def generate_synthetic(
    cfg: PipelineConfig,
    ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a realistic synthetic PPP count matrix for pipeline testing.
 
    Design principles
    -----------------
    • Negative binomial background mimics sparse RNA-seq read counts.
    • Signal genes drawn from PPP_GENESETS – each subtype has one dominant
      gene set upregulated with a Poisson signal layer.
    • Per-sample Gaussian noise simulates technical variability.
    • HDLSS regime by default (n=40, p=8000) – stresses dim-reduction
      and clustering stability methods.
 
    Returns
    -------
    expr        : DataFrame (genes * samples), integer counts
    true_labels : Series   (sample → true subtype name)
    """
    
    rng =np.random.default_rng(cfg.random_seed)
    n, p = cfg.synthetic_n_samples, cfg.synthetic_n_genes
    k = min(cfg.synthetic_n_subtypes, len(PPP_GENESETS))
    
    # Build gene list: PPP genes first, then background
    ppp_gene = get_all_ppp_genes()
    background = [f"GENE{i:05d}" for i in range(1, p + 1)]
    gene_names = list(dict.fromkeys(ppp_gene + background))[:p]

    # Sparse negative-binomial background (low-coverage RNA-seq)
    data = rng.negative_binomial(n=5, p=0.6, size=(p, n)).astype(float)
    
    subtype_names = list(PPP_GENESETS.keys())[:k]
    labels: list[str] = []
    samples_per = n // k
    col = 0
    
    for i, stype in enumerate(subtype_names):
        sig_genes = PPP_GENESETS[stype]
        sig_idx = [j for j, g in enumerate(gene_names) if g in sig_genes]
        n_this = samples_per if i < k - 1 else n - col
        
        for _ in range(n_this):
                        
            # strong signal in signature genes + Gaussian noise
            if sig_idx:
                data[sig_idx, col] += rng.poisson(lam=120, size=len(sig_idx))
                
            # Per-sample technical noise
            data[:, col] += np(rng.normal(0, 2, size=p), 0, None)
            labels.append(stype)
            col += 1
    
    sample_ids = [f"PPP_{i:03d}" for i in range(n)]
    expr = pd.DataFrame(
        data.clip(0).astype(int), 
        index=gene_names[:p],
        columns=sample_ids
    )
    true_labels = pd.Series(labels, index=sample_ids, name="true_subtype")
    
    logging.info(
        f"[Data] Synthetic: {expr.shape[0]} genes * {expr.shape[1]} samples" 
        f" | p/n ratio = {expr.shape[0]/n:.of} (HDLSS regime)"
    )
    
    logging.info(
        f"[Data] True subtype distribution: "
        f"{true_labels.value_counts().to_dict()}"
    )
    return expr, true_labels


 
# =============================================================================
# UNIFIED ENTRY POINT
# =============================================================================

def load_data(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load expression data according to cfg.use_geo.
 
    Returns
    -------
    expr        : DataFrame (genes * samples)
    true_labels : Series or None  (only for synthetic data)
    """
    if cfg.use_geo:
        expr = load_geo(cfg)
        return expr, None
    else:
        return generate_synthetic(cfg)
 