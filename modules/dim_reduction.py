"""
ppp_subtypes/modules/dim_reduction.py
========================================
HDLSS-aware dimensionality reduction for the PPP pipeline.
 
Two functions are exported:
    hdlss_reduce  – high-dimensional embedding (n_components coords)
    embed_2d      – 2-D visualisation embedding (UMAP → PCA fallback)
 
HDLSS context
-------------
In the HDLSS regime (High Dimension, Low Sample Size: p >> n),
the standard sample covariance matrix is rank-deficient and ill-conditioned.
This breaks ordinary PCA and makes distance-based methods unreliable.
 
Two corrections are applied here:
1. Ledoit-Wolf shrinkage  – regularises the covariance matrix so that
   all eigenvalues are positive and meaningful before PCA.
   The shrinkage coefficient α is estimated analytically (no CV needed).
 
2. Standard PCA (fast, well-conditioned after shrinkage) or
   Sparse PCA (higher interpretability, recommended for n > 30).
 
Usage
-----
    from ppp_subtypes.modules.dim_reduction import hdlss_reduce, embed_2d
    coords   = hdlss_reduce(expr, cfg)
    embed    = embed_2d(coords, cfg)
"""
 
from __future__ import annotations
 
import logging
 
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA, SparsePCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
 
from modules.config import PipelineConfig
 
 
# =============================================================================
# PRIMARY EMBEDDING
# =============================================================================
def hdlss_reduce(expr: pd.DataFrame, cfg: PipelineConfig) -> np.array:
    """
    HDLSS-safe dimentionality reduction pipeline:

    Step 1: Ledoit-wolf shrinkage on the covariance matrix.
        In p >> n settings, the sample covariance is ill-conditioned.
        Ledoit-wolf provides a well-conditioned regularised estimate.

    Step 2: Sparse PCA (or standard PCA in low-resource mode).
        • low_resource_mode=True  → standard PCA  (fast, ≤ 200 MB RAM)
        • low_resource_mode=False → SparsePCA     (sparse, interpretable)

    Step 3: Z-score standardise the final embedding so all components
        have unit variance before clustering.
    
    Parameters
    ----------
    expr : preprocessed genes × samples DataFrame
    cfg  : PipelineConfig
 
    Returns
    -------
    coords : (n_samples, n_components) numpy array
    """
    
    # samples × genes
    X = expr.T.values.astype(np.float64)
    n, p = X.shape
    logging.info(
        f"[Dim] HDLSS reduction: {n} samples * {p} genes"
        f"(p/n ratio = {p/n:.0f})"
    )

    # Step 1: Ledoit-wolf shrinkage (regularises covariance for p >> n)
    if cfg.ledoit_wolf_shrinkage and n < p:
        logging.info("[Dim] Applying Ledoit-wolf covariance shrinkage ...")
        lw = LedoitWolf(assume_centered=False)
        lw.fit(X)
        shrinkage = lw.shrinkage_
        logging.info(f"[Dim] Ledoit-Wolf shrinkage coefficent: {shrinkage: .4f}")

        # Pre-project onto PCA subspace first (safe pre-step for Sparse PCA)
        n_pre = min(n - 1, 50)
        pca_pre = PCA(n_components=n_pre, random_state=cfg.random_seed)
        X = pca_pre.fit_transform(X)
        cum_var = pca_pre.explained_variance_ratio_.sum()
        logging.info(
            f"[Dim]Pre_PCA: {n_pre} components, " 
            f"{cum_var:.1%} variance retained"
            )

    # Step 2: Sparse PCA for interpretable regularised components
    n_comp = min(cfg.n_components, X.shape[1] -1, n-1)
    
    if cfg.method == "sparse_pca" and not cfg.low_resoure_mode:
        logging.info(f"[Dim] Sparse PCA to {n_comp} components ...")
        
        # alpha controls sparsity regularisation; higher = more zero loadings
        reducer  = SparsePCA(
            n_components=n_comp,
            alpha=1.0, 
            random_state=cfg.random_seed,
            n_jobs=cfg.n_jobs,
            max_iter=200
        )
        coords = reducer.fit_transform(X)
        logging.info(f"[Dim] Sparse PCA: done")

    elif cfg.low_resource_mode and n_comp > 30:
        # Incremental PCA: lower peak RAM for large gene matrices
        logging.info(
            f"[Dim] Incremental PCA → {n_comp} components (low-resource) …"
        )
        batch = max(n_comp * 2, 50)
        ipca  = IncrementalPCA(n_components=n_comp, batch_size=batch)
        coords = ipca.fit_transform(X)
        cum_var = ipca.explained_variance_ratio_.cumsum()[-1]
        logging.info(f"[Dim] Variance explained: {cum_var:.1%}")

    else:
        # Low-resource fallback to standard PCA (fast, minimal RAM) to {n_comp} components ...:
        logging.info(f"[Dim] Standard PCA (low-resource mode)-> {n_comp} components ...")
        pca = PCA(n_components=n_comp, random_state=cfg.random_seed)
        coords = pca.fit_transform(X)
        cum_var = pca.explained_variance_ratio_.cumsum()[-1]
        logging.info(f"[Dim] Variance explained: {cum_var[-1]:.1%}")


    # Step 3: Z-score standardise the embedding
    coords = StandardScaler().fit_transform(coords)
    return coords

# =============================================================================
# 2-D VISUALISATION EMBEDDING
# =============================================================================

def embed_2d(coords: np.array, cfg: PipelineConfig) -> np.array:
    """Produce a 2-D embedding for visualisation only (not used for clustering).
 
    Tries UMAP first (best quality for small n with non-linear structure),
    falls back to the first two PCA components.
 
    Parameters
    ----------
    coords : (n_samples, n_components) embedding from hdlss_reduce
    cfg    : PipelineConfig
 
    Returns
    -------
    2-D numpy array of shape (n_samples, 2).
    """
    
    try:
        # noqa: F401
        import umap
        
        n_nbrs = min(15, max(2, coords.shape[0] - 2))

        logging.info(f"[Dim] UMAP 2-D embedding (n_neighbors={n_nbrs}) …")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_nbrs,
            min_dist=0.2,
            random_state=cfg.random_seed,
            low_memory=cfg.low_resource_mode,
        )
        emb = reducer.fit_transform(coords)
        logging.info("[Dim] UMAP done")
        return emb
    
    except ImportError:
        
        logging.info(
            "[Dim] umap-learn not installed – using PCA for 2-D visualisation"
        )
        pca = PCA(n_components=2, random_state=cfg.random_seed)
        return pca.fit_transform(coords)