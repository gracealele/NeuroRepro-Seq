from __future__ import annotations 
import logging
 
import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform
 
from modules.config import PipelineConfig
from modules.genesets import get_all_ppp_genes
 
 

# STEP 1 – Low-expression filter
def  filter_low_expression(
    expr: pd.DataFrame, 
    min_count: int, 
    min_frac: float
    ) -> pd.DataFrame:
    """
    Remove genes not expressed above min_count in at least min_frac of samples.
    """    
    n_samples = expr.shape[1]
    min_samples = max(1, int(min_frac * n_samples))
    expressed = (expr >= min_count).sum(axis=1) >= min_samples
    filtered = expr.loc[expressed]
    logging.info(
        f"[Pre] Low-expression filter: {expr.shape[0]} to {filtered.shape[0]} genes"
        f" (min count={min_count} in ≥{min_frac:.0%} of samples)"
    )
    return filtered


# STEP 2 – Normalisation
def tmm_normalise(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight TMM-inspired normalisation for low-coverage RNA-seq.
    Reference sample = closest to mean library size.
    Trim top/bottom 30% of log-ratios, scale, return log2(CPM+1).
    """
    
    lib_sizes = expr.sum(axis=0)
    ref_col = (lib_sizes - lib_sizes.mean()).abs().idxmin()
    
    # Avoid division by zero
    ref = expr[ref_col].replace(0, np.nan)
    
    norm_factors: dict[str, float] = {}
    for col in expr.columns:
        sample = expr[col].replace(0, np.nan)
        log_ratio = np.log2(sample / ref).dropna()
        
        if len(log_ratio) < 10:
            norm_factors[col] = 1.0
            continue
        
        # Trim top/bottom 30% to remove highly variable genes
        lo, hi = log_ratio.quantile(0.30), log_ratio.quantile(0.70)
        trimmed = log_ratio[(log_ratio >= lo) & (log_ratio <= hi)]
        norm_factors[col] = 2 ** trimmed.mean() if len(trimmed) > 0 else 1.0        

    nf = pd.Series(norm_factors)        
    normalised = expr.div(nf, axis=1)    
    cpm = normalised.div(normalised.sum(axis=0), axis=1) * 1e6
    log2_cpm = np.log2(cpm + 1)
    logging.info("[Pre] TMM normalisation → log2(CPM+1)")
    return log2_cpm

 
def quantile_normalise(expr: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    
    # Quantile normalisation — alternative to TMM for microarray data.
    n_q  = max(10, min(50, expr.shape[1]))
    norm = quantile_transform(
        expr.T.values, n_quantiles=n_q,
        output_distribution="normal",
        random_state=cfg.random_seed,
    )
    result = pd.DataFrame(norm.T, index=expr.index, columns=expr.columns)
    logging.info("[Pre] Quantile normalisation applied")
    return result


# STEP 3 – Variance filter (MAD)

def mad_variance_filter(
    expr: pd.DataFrame, 
    top_n: int, 
    ppp_genes: list[str], 
    ppp_weight:float,
    ) -> pd.DataFrame:
    """
    Select top_n genes by MAD. MAD is robust to outlier samples (critical for small n).
    PPP signature genes receive a score bonus to ensure biological relevance.
    """    
    
    # Median absolute deviation per gene
    medians = expr.median(axis=1)
    mad = (expr.subtract(medians, axis=0)).abs().median(axis=1)
            
    # PPP gene bonus        
    bonus = pd.Series(0.0, index=mad.index)        
    max_mad = mad.max()
    for g in ppp_genes:        
        if g in bonus.index:        
            bonus[g] = mad.max() * (ppp_weight  - 1.0)        
    
    scored = mad + bonus        
    top_genes = scored.nlargest(min(top_n, len(scored))).index
    
    filtered = expr.loc[top_genes]
    
    n_ppp_kept = sum(g in set(ppp_genes) for g in top_genes)
    logging.info(
        f"[Pre] MAD filter to {expr.shape[0]} -> {filtered.shape[0]} genes" 
        f"({n_ppp_kept} PPP signature genes retained"
        )        
    return filtered
              

# MAIN ENTRY POINT

def preprocess(expr_raw: pd.DataFrame, cfg:PipelineConfig) -> pd.DataFrame:        
    
    """
    Full preprocessing pipeline:
        1. Low-expression filter
        2. Normalisation (TMM or quantile)
        3. MAD variance filter with PPP gene upweighting
    """
    logging.info(
        f"[Pre] Input: {expr_raw.shape[0]} genes * {expr_raw.shape[1]} samples"
    )
    
    # 1. Low-expression filter
    expr = filter_low_expression(
        expr_raw, 
        min_count=cfg.min_count_threshold,
        min_frac=cfg.min_samples_expressed
    )
    
    # 2. Normalisation
    if cfg.normalization == "tmm":
        expr = tmm_normalise(expr)
    else:
        expr = quantile_normalise(expr, cfg)
    
    # 3. Variance filter with PPP gene priority
    ppp_genes = get_all_ppp_genes() if cfg.use_ppp_genesets else []
    expr = mad_variance_filter(
        expr,
        top_n=cfg.top_var_genes,
        ppp_genes=ppp_genes,
        ppp_weight=cfg.geneset_weighting,
    )
               
    logging.info(
        f"[Pre] Final matrix: {expr.shape[0]} genes * {expr.shape[1]} samples")
    return expr  
