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

def  filter_low_expression(expr: pd.DataFrame, min_count: int, min_frac: float) -> pd.DataFrame:
    """ Remove genes not expressed above threshold in at least min_frac of samples."""
    
    n_sample = expr.shape[1]
    mask = (expr >= min_count).sum(axis=1) >= int(min_frac * n_sample)
    filtered = expr.loc[mask]
    logging.info(f"Expression filter: {expr.shape[0]} to {filtered.shape[0]} genes")
    return expr, filtered


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
    return np.log2(cpm + 1)
        

def mad_variance_filter(expr: pd.DataFrame, top_n: int, 
                        ppp_genes: list, weight:float) -> pd.DataFrame:
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
            bonus[g] = mad.max() * (weight  - 1)        
    scored = mad + bonus        
    top_genes = scored.nlargest(top_n).index        
    logging.info(f"MAD filter to {len(top_genes)} genes selected " 
                 f"({sum(g in ppp_genes for g in top_genes)} PPP signature genes)")        
    return expr.loc[top_genes]
              

def preprocess(expr_raw: pd.DataFrame, cfg:PipelineConfig) -> pd.DataFrame:        
    expr = filter_low_expression(expr_raw, cfg.min_samples_expressed)        
    logging.info("Applying TMM normalization ...")        
    expr = tmm_normalise(expr)        
    ppp_genes = get_all_ppp_genes() if cfg.use_ppp_genesets else []
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
        logging.info(f"Pre_PCA: {n_pre} components, " 
                     f"{pca_pre.explained_variance_ratio_.sum():.1%} variance")

    # Step 2: Sparse PCA for interpretable regularised components
    n_comp = min(cfg.n_components, X.shape[1] -1, n-1)
    
    if cfg.method == "sparse_pca" and not cfg.low_resoure_mode:
        logging.info(f"Sparse PCA to {n_comp} components ...")
        
        # alpha controls sparsity; higher = more zero loadings
        spca = SparsePCA(n_components=n_comp, alpha=1.0, 
                         random_state=cfg.random_seed, n_jobs=cfg.n_jobs,
                         max_iter=200)
        coords = spca.fit_transform(X)
    else:
        # Low-resource fallback to standard PCA (fast, minimal RAM)
        logging.info(f"PCA (low-resource mode)-> {n_comp} components ...")
        pca = PCA(n_components=n_comp, random_state=cfg.random_seed)
        coords = pca.fit_transform(X)
        cum_var = pca.explained_variance_ratio_.cumsum()
        logging.info(f"Cumulative variance explained: {cum_var[-1]:.1%}")


    # Step 3: Z-score standardise the embedding
    coords = StandardScaler().fit_transform(coords)
    return coords

def embed_2d(coords: np.array, cfg: PipelineConfig) -> np.array:
    """2D embedding for visualization (UMAP to PCA fallback)."""
    try:
        import umap
        logging.info("Applying UMAP for 2D embedding ...")
        reducer = umap.UMAP(n_components=2, random_state=cfg.random_seed,
                            n_neighbors=min(15, coords.shape[0] - 1),
                            min_dist=0.2)
        return reducer.fit_transform(coords)
    except ImportError:
        logging.info("UMAP not available, falling back to PCA for 2D embedding ...")
        return PCA(n_components=2, random_state=cfg.random_seed).fit_transform(coords)



# =============================================================================
# 4. CONSENSUS CLUSTERING WITH BOOTSTRAP STABILITY
# =============================================================================


def _kmeans_once(coords: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=5, max_iter=200,
                  random_state=seed).fit_predict(coords)

def consensus_clustering(coords: np.ndarray, cfg: PipelineConfig) -> dict:
    """
    Consensus clustering with bootstrap stability assessment:

    1. For each k in k_range, repeat clustering n_iterations times on subsamples.
    2. Build consensus matrix: proportion of times each pair co-clustered.
    3. Assess stability: average consensus within clusters vs between clusters.
    4. Optionally, compute bootstrap confidence intervals for stability scores.
    5. Select optimal k based on stability and CDF heuristics.
    """
    n = coords.shape[0]
    results = {}
    
    for k in cfg.k_range:
        logging.info(f"Consensus clustering for k={k} ({cfg.n_iterations} iters) ...")
        co_occur = np.zeros((n, n), dtype=np.float32)
        co_sample = np.zeros((n, n), dtype=np.float32)
        
        for it in range(cfg.n_iterations):
            seed = cfg.random_seed + i
            idx = np.sort(np.random.choice(
                n, size=max(k + 1, int(n * cfg.subsample_rate)), replace=False))
 
            lbl = _kmeans_once(coords[idx], k, seed=it)
            
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i,j = idx[a], idx[b]
                    co_sample[i, j] += 1
                    co_sample[j, i] += 1
                    if lbl[a] == lbl[b]:
                        co_occur[i, j] += 1
                        co_occur[j, i] += 1
        
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.where(co_sample > 0, co_occur / co_sample, 0.0)
        np.file_diagonal(mat, 1.0)
        results[k] = mat
    
    return results

def bootstrap_stability(coords: np.ndarray, k: int,
                        cfg: PipelineConfig) -> float:

    """
    Jaccard-based cluster stability via bootstrap resampling.
    Returns mean Jaccard similarity across bootstrap replicates.
    Critical for HDLSS settings where cluster solutions can be unstable.
    """
    based_labels = _kmeans_once(coords, k, seed=cfg.random_seed)
    jaccard_scores = []
    
    for i in range(cfg.bootstrap_n):
        boot_idx = resample(np.arange(len(coords)), random_state=i) 
        boot_labels = _kmeans_once(coords[boot_idx], k, seed=i)
        
        # Align labels (best-match permutation via confusion)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(base_labels[boot_idx], boot_labels,
                               labels=list(range(k)))
        # Jaccard for each cluster pair
        row_scores = []
        for r in range(k):
            best = cm[r].max()
            union = cm[r].sum() + cm[:, cm[r].argmax()].sum() - best
            row_scores.append(best / union if union > 0 else 0)
        jaccard_scores.append(np.mean(row_scores))
        
    stability = float(np.mean(jaccard_scores))
    logging.info(f" Bootstrap stability for k={k}: {stability:.3f} " 
                 f"({'✓ stable' if stability >= cfg.stability_threshold else '✗ unstable'})")
    return stability


def  select_optimal_k(consensus_results: dict, 
                      coords: np.ndarray,
                      cfg: PipelineConfig) -> tuple[int, dict]:
    """
    Multi-criterion optimal k selection:
      1. Consensus CDF AUC (primary)
      2. Change (Δ) in AUC (elbow)
      3. Bootstrap stability filter
      4. Silhouette score (secondary)
 
    A k is only accepted if bootstrap stability ≥ threshold.
    """
    
    metrics = {}
    for k, mat in consensus_matrics.items():
        vals = mat[np.triu_indices_from(mat, k=1)]
        hist, edges = np.histogram(vals, bins = 100, range = (0, 1))
        cdf = np.cumsum(hist) / hist.sum()
        auc = float(np.trapezoid(cdf, edges[1:]) if hasattr(np, "trapezoid")
                    else np.trapez(cdf, edges[1:]))
        
        lbl = fcluster(linkage(squareform(np.clip(1 - mat, 0, 1)), method ="average"), k, criterion="maxclust") - 1
        sil = silhouette_score(coords, lbl) if len(set(lbl)) > 1 else - 1
        
        stab = (bootstrap_stability(coords, k, cfg)
                if cfg.bootstrap_ci else None)
        
        metrics[k] = {"auc": auc, "silhouette": sil, "stability": stab, "labels": lbl}
         
    
    k_vals = sorted(metrics)
    delta = {k_vals[i]: metrics[k_vals[i]]["auc"] - metrics[k_vals[i-1]]["auc"]
             
             for i in range(1, len(k_vals))}
    
    # Fliter by stability
    stable_ks = [k for k in delta
        if metrics[k]["stability"] is None
        or metrics[k]["stability"] >= cfg.stability_threshold]
    
    if not stable_ks:
        logging.warning(f"No k passed stability threshold, using best AUC")
        stable_ks = list(delta.keys())

    optimal_k = max(stable_ks, key=lambda k: delta[k])
    
    logging.info("\n── Cluster selection summary ──")
    for k in k_vals:
        m = metrics[k]
        s = f"  k={k}  AUC={m['auc']:.4f}  sil={m['silhouette']:.3f}"
        if m["stability"] is not None:
            s += f"  stab={m['stability']:.3f}"
        logging.info(s)
    logging.info(f"→ Optimal k = {optimal_k}\n")

    return optimal_k, metrics



def assign_subtypes(coords: np.ndarray, k: int,
                    sample_ids: pd.Index) -> pd.Series:
    lbl = _kmeans_once(coords, k, seed=42)
    letters = [f"Subtype_{chr(65+l)}" for l in lbl]
    return pd.Series(letters, index=sample_ids, name="predicted_subtype")
 
 
# =============================================================================
# 5. BIOLOGICAL CHARACTERISATION
# =============================================================================
 
def marker_genes_mwu(expr: pd.DataFrame,
                      subtypes: pd.Series,
                      top_n: int = 25) -> pd.DataFrame:
    """
    Mann-Whitney U test for each gene × subtype (one vs rest).
    MWU is non-parametric and appropriate for small, non-normal samples.
    Returns top_n marker genes per subtype with effect sizes.
    """
    logging.info("Identifying marker genes (Mann-Whitney U, non-parametric) …")
    marker_rows = []
    for st in sorted(subtypes.unique()):
        in_idx  = subtypes == st
        out_idx = ~in_idx
        for gene in expr.index:
            x_in  = expr.loc[gene, in_idx].values
            x_out = expr.loc[gene, out_idx].values
            if x_in.std() < 1e-6 and x_out.std() < 1e-6:
                continue
            stat, pval = mannwhitneyu(x_in, x_out, alternative="greater")
            # Rank-biserial correlation as effect size
            n1, n2 = len(x_in), len(x_out)
            effect = 1 - (2 * stat) / (n1 * n2)
            is_ppp = gene in get_all_ppp_genes()
            marker_rows.append({
                "subtype": st, "gene": gene,
                "mwu_stat": stat, "p_value": pval,
                "effect_size": effect, "is_ppp_gene": is_ppp,
                "mean_in": x_in.mean(), "mean_out": x_out.mean()
            })
 
    df = pd.DataFrame(marker_rows)
    # Sort by effect size; within ties, prioritise PPP genes
    df = df.sort_values(["subtype", "is_ppp_gene", "effect_size"],
                         ascending=[True, False, False])
    top = df.groupby("subtype").head(top_n)
    return top.reset_index(drop=True)
 
 
def geneset_enrichment_score(expr: pd.DataFrame,
                              subtypes: pd.Series) -> pd.DataFrame:
    """
    Simple mean-based enrichment of PPP gene sets per subtype.
    Provides a lightweight alternative to GSEA for low-resource settings.
    """
    rows = []
    for st in sorted(subtypes.unique()):
        sub_expr = expr.loc[:, subtypes == st]
        for gs_name, gs_genes in PPP_GENESETS.items():
            present = [g for g in gs_genes if g in sub_expr.index]
            score = sub_expr.loc[present].mean().mean() if present else np.nan
            rows.append({"subtype": st, "geneset": gs_name,
                          "mean_score": score, "n_genes_present": len(present)})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="geneset", columns="subtype", values="mean_score")
    return pivot
 
 
def pathway_report(subtypes: pd.Series,
                    markers: pd.DataFrame) -> dict:
    """Map dominant marker genes to PPP pathway hints."""
    report = {}
    for st in sorted(subtypes.unique()):
        st_markers = markers[markers["subtype"] == st]
        ppp_hits = st_markers[st_markers["is_ppp_gene"]]["gene"].tolist()
        matched_sets = []
        for gs_name, gs_genes in PPP_GENESETS.items():
            overlap = [g for g in ppp_hits if g in gs_genes]
            if overlap:
                matched_sets.append((gs_name, overlap, PATHWAY_HINTS[gs_name]))
        report[st] = matched_sets
    return report
 
 
# =============================================================================
# 6. VISUALISATION
# =============================================================================
 
PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#9B5DE5"]
STYLE = {"figure.facecolor": "white", "axes.facecolor": "#F8F8F8",
         "axes.spines.top": False, "axes.spines.right": False}
 
def _subtype_colors(subtypes: pd.Series) -> dict:
    return {s: PALETTE[i % len(PALETTE)]
            for i, s in enumerate(sorted(subtypes.unique()))}
 
 
def plot_consensus_heatmaps(matrices: dict, optimal_k: int, out: Path):
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1: axes = [axes]
    for ax, (k, mat) in zip(axes, sorted(matrices.items())):
        sns.heatmap(mat, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False,
                    cbar_kws={"shrink": 0.6, "label": "Co-cluster prob."})
        is_opt = k == optimal_k
        ax.set_title(f"k={k}" + (" ← optimal" if is_opt else ""),
                     fontweight="bold" if is_opt else "normal", fontsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor("#E63946" if is_opt else "none")
            sp.set_linewidth(3 if is_opt else 0)
    fig.suptitle("Consensus Matrices – PPP Subtype Discovery",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_embedding(embed: np.ndarray, subtypes: pd.Series,
                   true_labels: Optional[pd.Series], out: Path):
    ncols = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1: axes = [axes]
    cmap = _subtype_colors(subtypes)
 
    def _panel(ax, labels, title):
        for st, col in cmap.items():
            m = [l == st for l in labels]
            ax.scatter(embed[m, 0], embed[m, 1], c=col, label=st,
                       s=70, alpha=0.85, edgecolors="white", linewidths=0.5)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Dim-1"); ax.set_ylabel("Dim-2")
        ax.legend(fontsize=8, framealpha=0.7)
 
    _panel(axes[0], subtypes.tolist(), "Predicted Subtypes")
    if true_labels is not None:
        tcmap = _subtype_colors(true_labels)
        for st, col in tcmap.items():
            m = true_labels == st
            axes[1].scatter(embed[m.values, 0], embed[m.values, 1],
                            c=col, label=st, s=70, alpha=0.85,
                            edgecolors="white", linewidths=0.5)
        axes[1].set_title("True Subtypes (synthetic)", fontweight="bold", fontsize=12)
        axes[1].set_xlabel("Dim-1"); axes[1].set_ylabel("Dim-2")
        axes[1].legend(fontsize=8, framealpha=0.7)
 
    fig.suptitle("Embedding – PPP Molecular Subtypes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_marker_heatmap(expr: pd.DataFrame, subtypes: pd.Series,
                         markers: pd.DataFrame, out: Path, top_n: int = 15):
    top_genes = (markers.groupby("subtype")
                 .apply(lambda d: d.nlargest(top_n, "effect_size")["gene"])
                 .explode().unique().tolist())
    top_genes = [g for g in top_genes if g in expr.index][:top_n * len(subtypes.unique())]
 
    sub_expr = expr.loc[top_genes, subtypes.sort_values().index]
    cmap = _subtype_colors(subtypes)
    col_colors = subtypes[sub_expr.columns].map(cmap)
 
    fig, ax = plt.subplots(figsize=(max(10, len(sub_expr.columns) * 0.22 + 4),
                                    max(8, len(top_genes) * 0.32 + 2)))
    sns.heatmap(sub_expr, ax=ax, cmap="RdBu_r", center=0,
                xticklabels=False, yticklabels=True,
                cbar_kws={"label": "Log2 CPM", "shrink": 0.6})
 
    handles = [mpatches.Patch(color=c, label=s) for s, c in cmap.items()]
    ax.legend(handles=handles, title="Subtype", loc="upper right",
              bbox_to_anchor=(1.18, 1), fontsize=8)
    ax.set_title("Marker Genes per Subtype (MWU, effect size ranked)",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Samples"); ax.set_ylabel("Marker Genes")
 
    # Mark PPP signature genes with a star
    ppp_set = set(get_all_ppp_genes())
    for label in ax.get_yticklabels():
        if label.get_text() in ppp_set:
            label.set_color("#E63946")
            label.set_fontweight("bold")
 
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_k_selection(metrics: dict, optimal_k: int, out: Path):
    ks   = sorted(metrics)
    aucs = [metrics[k]["auc"]        for k in ks]
    sils = [metrics[k]["silhouette"] for k in ks]
    stbs = [metrics[k]["stability"]  for k in ks]
 
    has_stab = all(s is not None for s in stbs)
    nrows = 3 if has_stab else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(6, 3 * nrows), sharex=True)
 
    def _line(ax, y, label, colour, ylabel, hline=None):
        ax.plot(ks, y, "o-", color=colour, linewidth=2.2, markersize=8, label=label)
        ax.axvline(optimal_k, color="#E63946", linestyle="--", linewidth=1.5,
                   label=f"Optimal k={optimal_k}")
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle=":", linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
 
    _line(axes[0], aucs, "Consensus AUC", "#457B9D", "CDF AUC")
    _line(axes[1], sils, "Silhouette",    "#2A9D8F", "Silhouette Score", hline=0)
    if has_stab:
        _line(axes[2], stbs, "Bootstrap Stability", "#9B5DE5",
              "Stability (Jaccard)", hline=0.75)
 
    axes[-1].set_xlabel("Number of clusters (k)", fontsize=10)
    fig.suptitle("Optimal k Selection – Multi-criterion", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_geneset_scores(enrichment: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    enrichment_norm = enrichment.div(enrichment.max(axis=1), axis=0)
    sns.heatmap(enrichment_norm, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                cbar_kws={"label": "Relative enrichment score"})
    ax.set_title("PPP Gene Set Enrichment per Subtype", fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted Subtype"); ax.set_ylabel("PPP Gene Set")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_compute_profile(profile_df: pd.DataFrame, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(profile_df))]
 
    ax1.barh(profile_df["stage"], profile_df["time_s"], color=colors)
    ax1.set_xlabel("Time (seconds)"); ax1.set_title("Runtime per Stage", fontweight="bold")
 
    ax2.barh(profile_df["stage"], profile_df["peak_ram_mb"], color=colors)
    ax2.set_xlabel("Peak RAM (MB)"); ax2.set_title("Memory per Stage", fontweight="bold")
 
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
# =============================================================================
# 7. REPORT
# =============================================================================


def write_report(cfg: PipelineConfig, subtypes: pd.Series,
                 metrics: dict, optimal_k: int,
                 markers: pd.DataFrame, pathway_rep: dict,
                 enrichment: pd.DataFrame, profile: pd.DataFrame,
                 out: Path):
    lines = [
        "=" * 68,
        "  PPP Molecular Subtype Pipeline v2.0 – Analysis Report",
        "=" * 68,
        "",
        "CONFIGURATION",
        f"  Samples            : {subtypes.shape[0]}",
        f"  Data mode          : {'GEO ' + cfg.geo_id if cfg.use_geo else 'Synthetic'}",
        f"  Normalisation      : {cfg.normalisation.upper()}",
        f"  Variance filter    : MAD (top {cfg.top_var_genes} genes)",
        f"  Dim. reduction     : {cfg.method} + Ledoit-Wolf shrinkage",
        f"  Clustering         : Consensus k-means ({cfg.n_iterations} iters)",
        f"  Stability filter   : Bootstrap Jaccard ≥ {cfg.stability_threshold}",
        "",
        "SUBTYPE DISCOVERY",
        f"  Optimal k          : {optimal_k}",
    ]
    for k in sorted(metrics):
        m = metrics[k]
        s = (f"  k={k}  AUC={m['auc']:.4f}  sil={m['silhouette']:.3f}"
             + (f"  stab={m['stability']:.3f}" if m['stability'] else ""))
        lines.append(s)
    lines += ["", "SUBTYPE DISTRIBUTION"]
    for st, n in subtypes.value_counts().items():
        lines.append(f"  {st}: {n} samples ({100*n/len(subtypes):.1f}%)")
 
    lines += ["", "TOP MARKER GENES (MWU, effect size)"]
    for st in sorted(subtypes.unique()):
        top5 = (markers[markers["subtype"] == st]
                .nlargest(5, "effect_size")[["gene", "effect_size", "is_ppp_gene"]])
        lines.append(f"\n  {st}:")
        for _, row in top5.iterrows():
            tag = " ★PPP" if row["is_ppp_gene"] else ""
            lines.append(f"    {row['gene']:<12} ES={row['effect_size']:.3f}{tag}")
 
    lines += ["", "PATHWAY ENRICHMENT HINTS"]
    for st, hits in pathway_rep.items():
        lines.append(f"\n  {st}:")
        if hits:
            for gs_name, genes, pathways in hits:
                lines.append(f"    [{gs_name}]  genes: {', '.join(genes[:4])}")
                for pw in pathways[:3]:
                    lines.append(f"      → {pw}")
        else:
            lines.append("    No PPP gene set overlap detected")
 
    lines += ["", "COMPUTE PROFILE"]
    for _, row in profile.iterrows():
        lines.append(f"  {row['stage']:<35} {row['time_s']:>6.2f}s  "
                     f"{row['peak_ram_mb']:>6.1f} MB RAM")
 
    total_t = profile["time_s"].sum()
    total_m = profile["peak_ram_mb"].max()
    lines += [
        f"\n  Total runtime      : {total_t:.1f}s",
        f"  Peak RAM           : {total_m:.1f} MB",
        "",
        "NOTE: Designed for low-resource clinical settings (laptop-compatible).",
        "      PPP signature genes (★) are from published literature.",
        "=" * 68,
    ]
    out.write_text("\n".join(lines))
    logging.info(f"Report → {out}")
 
 
# =============================================================================
# MAIN PIPELINE
# =============================================================================

 
def run(cfg: PipelineConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(exist_ok=True)
 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "pipeline.log", mode="w"),
        ],
    )
    np.random.seed(cfg.random_seed)
    profiler = Profiler(active=cfg.profile_runtime or cfg.profile_memory)
 
    logging.info("=" * 60)
    logging.info("  PPP Molecular Subtype Pipeline  v2.0")
    logging.info("=" * 60)
 
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
# ENTRY POINT
# =============================================================================
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPP Molecular Subtype Pipeline v2.0")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (optional)")
    parser.add_argument("--geo", default=None,
                        help="GEO accession (e.g. GSE152795) – overrides config")
    args = parser.parse_args()
 
    if args.config:
        cfg = PipelineConfig.from_json(args.config)
    else:
        cfg = PipelineConfig()
 
    if args.geo:
        cfg.use_geo = True
        cfg.geo_id  = args.geo
 
    run(cfg)
 
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