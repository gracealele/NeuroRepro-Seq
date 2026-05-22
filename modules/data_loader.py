

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
    
    # Download a GEO series → (genes × samples) DataFrame. Requires GEOparse.
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



# SYNTHETIC DATA

def generate_synthetic(
    cfg: PipelineConfig,
    ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Realistic synthetic PPP count matrix anchored to PPP gene signatures.
    HDLSS regime by default (n=40, p=8000).
    """
    
    rng =np.random.default_rng(cfg.random_seed)
    n, p = cfg.synthetic_n_samples, cfg.synthetic_n_genes
    k = min(cfg.synthetic_n_subtypes, len(PPP_GENESETS))
    
    # Build gene list: PPP genes first, then background
    ppp_gene = get_all_ppp_genes()
    background = [f"GENE{i:05d}" for i in range(1, p + 1)]
    gene_names = list(dict.fromkeys(ppp_gene + background))[:p]

    # Sparse negative-binomial background (mimics low-coverage RNA-seq)
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
            data[:, col] += np.clip(rng.normal(0, 2, size=p), 0, None)
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
        f" | p/n ratio = {expr.shape[0]/n:.0f} (HDLSS regime)"
    )
    
    logging.info(
        f"[Data] True subtype distribution: "
        f"{true_labels.value_counts().to_dict()}"
    )
    return expr, true_labels


 
# UNIFIED ENTRY POINT

def load_data(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    
    """Unified entry point: returns (expr, true_labels). true_labels is None for GEO."""    
    if cfg.use_geo:
        return load_geo(cfg), None
    return generate_synthetic(cfg)
 