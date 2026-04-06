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

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import SparsePCA, PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")

#---- Local modules ----
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ppp_genesets import PPP_GENESETS, PATHWAY_HINTS, get_all_ppp_genes
except ImportError:
    PPP_GENESETS = {}
    PATHWAY_HINTS = {}
    def get_all_ppp_genes():
        return []


# ================================================
# CONFIGURATION
# ================================================

@dataclasses.dataclass
class PipelineConfig:
    # Data
    use_geo: bool = False
    geo_id: str = "GSE152795"
    geo_cache_dir: str = "geo_cache"
    synthetic_n_samples: int = 40
    synthetic_n_genes: int = 8_000
    synthetic_n_subtypes: int = 3
    min_count_threshold: int = 5
    min_samples_expressed: float = 0.2
    
    # Preprocessing
    normalization: str = "tmm"  # "tmm" or "log2"
    top_var_genes: int = 1_000
    variance_method: str = "mad"  # "mad" or "var"
    
    # Dimensionality Reduction
    method: str = "sparse_pca"  # "pca" or "sparse_pca"
    n_components: int = 20
    ledoit_wolf_shrinkage: bool = True
    max_ram_mb: int = 512
    
    # Clustering
    k_range: list = dataclasses.field(default_factory=lambda: list(range(2, 7)))
    n_iterations: int = 100
    subsample_rate: float = 0.7
    bootstrap_ci: bool = True
    bootstrap_n: int = 50
    stability_threshold: float = 0.75
    
    # Biological Prioritization
    use_ppp_genesets: bool = True
    geneset_weighting: float = 2.0
    run_pathway_hints: bool = True
    
    # Compute
    profile_runtime: bool = True
    profile_memory: bool = True
    n_jobs: int = 1                     # None = auto-detect, 1 = no parallelism
    random_seed: int = 42
    low_resource_mode: bool = True
    
    # Output
    output_dir: str = "ppp_subtype_results"
    save_intermediate: bool = True
    report: bool = True
    
    @classmethod
    def from_json(cls, path: str) -> PipelineConfig:
        with open(path) as f:
            raw = json.load(f)
        flat = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
        field = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in flat.items() if k in field})
    
    
# ===============================================================
# PROFILE DECORATOR
# ===============================================================

class Profiler:
    """Lightweight runtime and memory profiler using time and tracemalloc."""

    def __init__(self, active: bool = True):
        self.active = active
        self._log: list[dict] = []
        
    def __call__(self, stage: str):
        return self._Stage(self, stage)
    
    class _Stage:
        def __init__(self, parent: Profiler, name: str):
            self.p = parent
            self.name = name
            
        def __enter__(self):
            self.t0 = time.perf_counter()
            if self.p.active:
                tracemalloc.start()
            return self
                
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self.t0
            mem_mb = 0.0
            if self.p.active:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_mb = peak / 1e6
            self.p._log.append({
                "stage": self.name, "time_s": round(elapsed, 3),
                "peak_ram_mb": round(mem_mb, 1)
            })
            logging.info(f" {self.name}: {elapsed:.2f}s, | RAM peak: {mem_mb:.1f}MB")  
    
    def summmary(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)


# =================================================================== 
# DATA LOADING
# ===================================================================

def load_geo(geo_id: str, cache_dir: str) -> pd.DataFrame:
    """Load gene expression data from GEO using GEOparse, with caching."""
    try:
        import GEOparse
    except ImportError:
        raise ImportError("Please install GEOparse: pip install GEOparse")

    logging.info(f"Downloading GEO dataset {geo_id} from GEO ...")
    gse = GEOparse.get_GEO(geo=geo_id, destdir=cache_dir, silent=True)
    frames = []
    for name, gsm in gse.gsms.items():
        if gsm.table is not None and not gsm.table.empty:
            col = gsm.table.set_index(gsm.table.columns[0])["VALUE"]
            col.name = name
            frames.append(col)
            
    # Extract expression data (assuming it's in the first GSM)
    expr = pd.concat(frames, axis=1).dropna()
    expr.index.name = "gene_id"
    logging.info(f"GEO loaded dataset: {expr.shape[0]} probes * {expr.shape[1]} samples")
    return expr

def generate_synthetic(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Realistic synthetci PPP count matrix.
    Three subtypes anchored to literure gene sets. 
    Designed for HDLSS regime (n=40 samples, p=8000 genes by default)
    """
    rng =np.random.default_rng(cfg.random_seed)
    n, p = cfg.synthetic_n_samples, cfg.synthetic_n_genes
    k = cfg.synthetic_n_subtypes
    
    ppp_gene = get_all_ppp_genes()
    extra = [f"GENE{i:05d}" for i in range(p)]
    gene_names = (ppp_gene + extra)[:p]

    # Negative binomial background (sparse, low-coverage)
    data = rng.negative_binomial(n=5, p=0.6, size=(p, n)).astype(float)
    
    subtype_names = list(PPP_GENESETS.keys())[:k]
    labels = []
    samples_per = n // k
    
    col = 0
    for i, stype in enumerate(subtype_names):
        sig_genes = PPP_GENESETS[stype]
        sig_idx = [j for j, g in enumerate(gene_names) if g in sig_genes]
        for _ in range(samples_per if i < k - 1 else n - col):
            
            # strong signal in signature genes + Gaussian noise
            if sig_idx:
                data[sig_idx, col] += rng.poisson(lam=120, size=len(sig_idx))
            data[:, col] += rng.normal(0, 2, size=p).clip(0)
            labels.append(stype)
            col += 1
    
    sample_ids = [f"PPP_{i:03d}" for i in range(n)]
    expr = pd.DataFrame(data.clip(0).astype(int), index=gene_names, columns=sample_ids)
    true_labels = pd.Series(labels, index=sample_ids, name="true_subtype")
    logging.info(f"Synthetic data: {expr.shape} | subtypes: {true_labels.value_counts().to_dict()}")
    return expr, true_labels


# =================================================================== 
# PREPROCESSING (HDLSS + low-resource aware)
# ===================================================================

def filter_low_expression(expr: pd.DataFrame, min_count: int, min_frac: float) -> pd.DataFrame:
    """ Remove genes not expressed above threshold in at least min_frac of samples."""
    
    n_sample = expr.shape[1]
    mask = (expr >= min_count).sum(axis=1) >= int(min_frac * n_sample)
    filtered = expr.loc[mask]
    logging.info(f"Expression filter: {expr.shape[0]} to {filtered.shape[0]} genes")
    return filtered
def tmm_normalise(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight TMM-inspired normalisation for low-coverage RNA-seq. Scales each sample 
    by a trimmed-mean factor relative to a referance sample (sample closest to the overall 
    mean libarary size). Robust to outlier genes and suitable for sparse count data.
    """
    lib_sizes = expr.sum(axis=0)
    ref_idx = (lib_sizes - lib_sizes.mean()).abs().idxmin()
    ref = expr[ref_idx].replace(0, np.nan)  # Avoid division by zero
    
    norm_factors = {}
    for col in expr.columns:
        s = expr[col].replace(0, np.nan)
        log_ratios = np.log2(s / ref).dropna()
        
        # Trim top/bottom 30% to remove highly variable genes
        lo, hi = log_ratios.quantile(0.30), log_ratios.quantile(0.70)
        trimmed = log_ratios[(log_ratios >= lo) & (log_ratios <= hi)]
        norm_factors[col] = 2 ** trimmed.mean() if len(trimmed) > 0 else 1.0        

    nf = pd.Series(norm_factors)        
    normalised = expr.div(nf, axis=1)    
    cpm = normalised.div(normalised.sum(axis=0), axis=1) * 1e6
    return np.log2(cpm +1)
        

def mad_variance_filter(expr: pd.DataFrame, top_n: int, ppp_genes: list, weight: float) -> pd.DataFrame:
    """
    Select top-n genes by Median Absolute Deviation (MAD).
    MAD is robust to outliers samples, this is critical in small-n HDLSS settings.        
    PPP signature gene receive a score bonus to ensure biological relevance.
    """    
    mad = expr.subtract(expr.median(axis=1), axis=0).abs().median(axis=1)
            
    # Upweigt PPP genes        
    bonus = pd.Series(0.0, index=mad.index)        
    for g in ppp_genes:        
        if g in bonus.index:        
            bonus[g] = mad.max() * (weight - 1)        
    scored = mad + bonus        
    top_genes = scored.nlargest(top_n).index        
    logging.info(f"MAD filter to {len(top_genes)} genes selected ({sum(g in ppp_genes for g in top_genes)} PPP signature genes)")        
    return expr.loc[top_genes]
              

def preprocess(expr_raw: pd.DataFrame, cfg:PipelineConfig) -> pd.DataFrame:        
    expr = filter_low_expression(expr_raw, cfg.min_samples_expressed)        
    logging.info("Applying TMM normalization ...")        
    expr = tmm_normalise(expr)        
    ppp_genes = get_all_pppp_genes() if cfg.use_ppp_genesets else []
    expr = mad_variance_filter(expr, cfg.top_var_genes, ppp_genes, cfg.genesets_weighting)
    return expr        


# =================================================================== 
# HDLSS-Aware DIMENTIONALITY REDUCTION
# ===================================================================

def hdlss_reduce(expr: pd.DataFrame, cfg: PipelineConfig) -> np.array:
    """
    HDLSS-safe dimentionality reduction pipeline:

    Step 1: Ledoit-wolf shrinkage on thr covariance matrix.
            In p >> n settings, the sample covariance is ill-conditioned.
            Ledoit-wolf provides a well-conditioned regularised estimate.

    Step 2: Sparse PCA (or standard PCA in low-resource mode).
            Sparse PCA finds components with few non-zero loadings.
            improving interpretability and reducing overfitting in HDLSS.

    Step 3: Z-score standardise the resulting embedding.
    """
    X = expr.T.values.astype(float)
    n, p = X.shape
    logging.info(f"HDLSS reduction: {n} samples * {p} genes(p/n ratio = {p/n:.0f})")

    # Step 1: Ledoit-wolf shrinkage (regularises covariance for p >> n)
    if cfg.ledoit_wolf_shrinkage and n < p:
        logging.info("Applying Ledoit-wolf covariance shrinkage ...")
        lw = LedoitWolf(assume_centered=False)
        lw.fit(X)
        shrinkage = lw.shrinkage_
        logging.info(f"Ledoit-Wolf shrinkage coefficent: {shrinkage: .4f}")

        # Project onto PCA space first (safe pre-step for Sparse PCA)
        n_pre = min(n - 1, 50)
        pca_pre = PCA(n_components=n_pre, random_state=cfg.random_seed)
        X = pca_pre.fit_transform(X)
        logging.info(f"Pre_PCA: {n_pre} components, {pca_pre.explained_variance_ratio_.sum():.1%} variance")

    # Step 2: Sparse PCA for interpretable regularised components
    n_comp = min(cfg.n_components, X.shape[1] -1, n-1)
    
    if cfg.method == "sparse_pca" and not cfg.low_resource_mode:
        logging.info(f"Sparse PCA to {n_comp} components ...")
        
        # alpha controls sparsity; higher = more zero loadings
        spca = SparsePCA(n_components=n_comp, alpha=1.0, 
                         random_state=cfg.random_seed, n_jobs=cfg.n_jobs,
                         max_iter=200)
        coords = spca.fit_transform(X)
    else:
        # Low-resource fallback to standard PCA (fast, minimal RAM)
        logging.info(f"Standard PCA to {n_comp} components (low-resource mode) ...")
        pca = PCA(n_components=n_comp, random_state=cfg.random_seed)
        coords = pca.fit_transform(X)
    
    # Z-score standardize the embedding
    scaler = StandardScaler()
    coords = scaler.fit_transform(coords)
    return coords

"""
Pipeline:
    1. Data loading: GEO download (GEOparse) OR synthetic fallback
    2. Preprocessing:  log2, quantile normalization, variance filtering
    3. Dimentionality Reduction: PCA to UMAP
    4. Consensus Clustering: Repeat k-means across k=2..6
    4. Optimal-k Selection: Consensus CDF + change or difference in CDF heuristic
    5. Subtype Characterization: Top marker genes, enrichment hints
    6. Visualisation: Consensus heatmap, UMAP coloured by subtype, markers
"""