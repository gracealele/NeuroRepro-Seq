# ppp-subtypes

**Molecular Disease Subtype Discovery Pipeline for Postpartum Psychosis**

A computationally efficient transcriptomic pipeline for identifying molecular disease subtypes in postpartum psychosis (PPP), specifically designed for low-resource clinical environments where sample sizes are small and sequencing infrastructure is limited.

---

## Overview

Postpartum psychosis affects approximately 1–2 per 1,000 deliveries and remains poorly characterised at the molecular level, partly due to the rarity of the condition and the consequent scarcity of large transcriptomic datasets. This pipeline addresses that challenge directly:

- **HDLSS-aware**: Built for the High Dimension, Low Sample Size regime (p >> n, e.g. n=20–80 patients, p=5,000–20,000 genes). Uses Ledoit-Wolf covariance shrinkage and bootstrap stability validation to ensure results are reproducible even at very small n.
- **Computationally efficient**: full pipeline runs in under 2 minutes on a standard laptop using less than 500 MB RAM. Every stage is profiled for runtime and memory.
- **Low-resource ready**: TMM normalisation handles sparse, noisy count data from low-coverage sequencing. A low-resource mode caps memory usage and avoids GPU dependencies entirely.
- **PPP-specific biology**: Four literature-curated gene signatures (neuroinflammatory, HPA/hormonal, dopaminergic/synaptic, immune-oxidative) are embedded in the pipeline and used to biologically annotate discovered subtypes.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/gracealele/ppp-subtypes.git
cd ppp-subtypes

# Install core dependencies
pip install -r requirements.txt

# Editable install (recommended for development)
pip install -e .

# With optional extras
pip install -e ".[geo]"    # enables live GEO data download
pip install -e ".[umap]"   # enables UMAP visualisation (better 2-D plots)
pip install -e ".[full]"   # all optional dependencies
```

**Requirements:** Python ≥ 3.10, pip

---

## Quick Start

### Using synthetic data (no data needed)

```python
from ppp_subtypes import run, PipelineConfig

cfg = PipelineConfig(
    synthetic_n_samples = 40,    # number of patient samples
    k_range             = [2, 3, 4, 5],
    n_iterations        = 100,
    low_resource_mode   = True,
)
results = run(cfg)

# Results dictionary contains:
#   results["subtypes"]   – pd.Series: predicted subtype per sample
#   results["markers"]    – pd.DataFrame: top marker genes per subtype
#   results["enrichment"] – pd.DataFrame: PPP gene-set scores per subtype
#   results["optimal_k"]  – int: chosen number of subtypes
#   results["profile"]    – pd.DataFrame: runtime + RAM per stage
```

### Using real GEO data

```python
cfg = PipelineConfig(
    use_geo = True,
    geo_id  = "GSE152795",   # postpartum mood/psychosis transcriptomics
)
results = run(cfg)
```

Recommended GEO accessions for PPP:

| Accession | Description |
|-----------|-------------|
| GSE152795 | Postpartum mood/psychosis transcriptomics |
| GSE116137 | Bipolar disorder first-episode (related) |
| GSE181797 | Peripheral blood in perinatal psychiatric disorders |
| GSE54913  | Postpartum depression transcriptomics |

### Using your own CSV data

```python
import pandas as pd
from ppp_subtypes import run, PipelineConfig
from modules.preprocessing import preprocess
from modules.dim_reduction import hdlss_reduce, embed_2d
from modules.clustering import consensus_cluster, select_optimal_k, assign_subtypes

# Load your expression matrix (genes x samples)
expr_raw = pd.read_csv("your_expression_data.csv", index_col=0)

cfg  = PipelineConfig(out_dir="my_results")
expr = preprocess(expr_raw, cfg)
# ... continue with remaining pipeline steps
```

### Command line

```bash
# Synthetic data (default)
python -m ppp_subtypes.main

# Custom config file
python -m ppp_subtypes.main --config config.json

# Real GEO data
python -m ppp_subtypes.main --geo GSE152795

# Adjust sample count and k range
python -m ppp_subtypes.main --n-samples 60 --k-max 5 --out my_results/
```

---

## Configuration

All pipeline behaviour is controlled through `PipelineConfig`. Key parameters:

```python
PipelineConfig(
    # Data
    synthetic_n_samples   = 40,      # number of synthetic patient samples
    synthetic_n_genes     = 8_000,   # gene pool size
    synthetic_n_subtypes  = 3,       # true subtypes in synthetic data

    # Preprocessing
    normalisation         = "tmm",   # "tmm" (RNA-seq) or "quantile" (microarray)
    top_var_genes         = 1_000,   # genes kept after MAD variance filter
    geneset_weighting     = 2.0,     # PPP gene priority boost in variance filter

    # Dimensionality reduction
    n_components          = 20,      # PCA components
    ledoit_wolf_shrinkage = True,    # regularise covariance (critical for HDLSS)
    low_resource_mode     = True,    # use standard PCA; cap RAM usage

    # Clustering
    k_range               = [2, 3, 4, 5],  # cluster numbers to test
    n_iterations          = 100,            # consensus resampling iterations
    subsample_rate        = 0.70,           # fraction of samples per iteration
    bootstrap_n           = 50,             # stability validation replicates
    stability_threshold   = 0.60,           # minimum Jaccard to accept a k

    # Output
    out_dir               = "ppp_results",
    write_report          = True,
    profile_runtime       = True,
)
```

You can also load from JSON:

```json
{
  "data":       { "synthetic_n_samples": 60, "use_geo": false },
  "clustering": { "k_range": [2, 3, 4], "n_iterations": 200 },
  "output":     { "out_dir": "results/run1", "write_report": true }
}
```

```bash
python -m ppp_subtypes.main --config config.json
```

---

## Pipeline Architecture

```
Raw counts / GEO data
        │
        ▼
1. data_loader.py      – load GEO or generate synthetic PPP data
        │
        ▼
2. preprocessing.py    – low-expression filter → TMM normalisation
                         → MAD variance filter (PPP gene priority)
        │
        ▼
3. dim_reduction.py    – Ledoit-Wolf shrinkage → PCA → standardise
                         (embed_2d for visualisation only)
        │
        ▼
4. clustering.py       – vectorised consensus clustering (k = 2..5)
                         → bootstrap Jaccard stability per k
                         → multi-criterion optimal k selection
        │
        ▼
5. characterisation.py – Mann-Whitney U marker genes (non-parametric)
                         → PPP gene-set enrichment scoring
                         → pathway annotation report
        │
        ▼
6. visualisation.py    – consensus heatmaps, embedding, marker heatmap,
                         k-selection plot, enrichment heatmap, profiler chart
        │
        ▼
7. reporter.py         – plain-text analysis report (clinical-ready)
```

---

## PPP Gene Signatures

Four literature-grounded gene sets are embedded in `genesets.py`:

| Signature | Key genes | Biological relevance |
|-----------|-----------|---------------------|
| **Neuroinflammatory** | IL1B, IL6, TNF, NFKB1, CRP, C3 | Cytokine dysregulation, neuroinflammation (Bergink 2015, Frey 2022) |
| **HPA_Hormonal** | ESR1, CRH, OXTR, PRL, NR3C1, FKBP5 | HPA axis, oestrogen/oxytocin signalling (Payne 2020, Deschamps 2016) |
| **Dopaminergic_Synaptic** | DRD2, BDNF, GRIN2B, SLC6A4, COMT | Dopamine/glutamate/serotonin circuits (Forty 2014, Bergink 2011) |
| **Immune_Oxidative** | SOD1, NFE2L2, FOXP3, HLA-DRA, GPX1 | Oxidative stress, T-cell regulation (Balan 2019, Dahan 2021) |

---

## Outputs

After a pipeline run, the output directory contains:

```
ppp_results/
├── sample_subtypes.csv          – predicted subtype label per sample
├── marker_genes.csv             – top MWU marker genes per subtype
├── geneset_enrichment.csv       – PPP pathway scores per subtype
├── compute_profile.csv          – runtime + RAM per stage
├── pipeline_config.json         – full config used for this run
├── consensus_heatmaps.png       – co-clustering matrices for each k tested
├── embedding.png                – 2-D sample embedding coloured by subtype
├── marker_heatmap.png           – expression heatmap of top marker genes
├── k_selection.png              – AUC, silhouette, stability vs k
├── geneset_enrichment.png       – PPP pathway enrichment heatmap
├── compute_profile.png          – runtime + RAM bar charts
├── profiler_log.csv             – detailed profiler log
├── profiler_timing.png          – profiler visualisation
├── report.txt                   – full plain-text analysis report
└── pipeline.log                 – complete run log
```

---

## Running Tests

```bash
# Run all 62 unit tests
python -m unittest ppp_subtypes/tests/test_pipeline.py -v

# Run a specific test class
python -m unittest ppp_subtypes.tests.test_pipeline.TestClustering -v

# Run with discovery
python -m unittest discover -s ppp_subtypes/tests -v
```

Tests cover all 10 modules independently and include two integration tests:
- Full pipeline end-to-end with all output files verified
- RAM usage validated under 500 MB

---

## Project Structure

```
ppp_subtypes/
│
├── __init__.py                  – public API (run, PipelineConfig, PPP_GENESETS)
├── main.py                      – pipeline runner + CLI entry point
│
├── modules/
│   ├── __init__.py              – module-level public API
│   ├── config.py                – PipelineConfig dataclass + JSON I/O
│   ├── profiler.py              – context-manager runtime/RAM profiler
│   ├── genesets.py              – PPP literature gene signatures
│   ├── data_loader.py           – GEO download + synthetic data generation
│   ├── preprocessing.py         – TMM normalisation, MAD filter
│   ├── dim_reduction.py         – Ledoit-Wolf + PCA/SparsePCA + UMAP
│   ├── clustering.py            – vectorised consensus clustering + stability
│   ├── characterisation.py      – MWU marker genes + gene-set enrichment
│   ├── visualisation.py         – all 6 plot functions
│   └── reporter.py              – plain-text analysis report writer
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         – 62 unit tests (stdlib unittest, no pytest)
│
├── database/                    – local data files (not tracked by git)
│   └── .gitkeep
│
├── README.md
├── requirements.txt
└── setup.py
```

---

## Roadmap

- [ ] Wire `database/` CSV files into `data_loader.py` as a local data source
- [ ] DESeq2-style differential expression between subtypes
- [ ] Sparse PCA mode for fully interpretable gene loadings
- [ ] Survival / clinical outcome association per subtype
- [ ] Web-based result viewer

---

## Citation

If you use this pipeline in your research, please cite:

> Alele, G. (2025). *ppp-subtypes: Molecular Disease Subtype Discovery Pipeline
> for Postpartum Psychosis*. GitHub. https://github.com/gracealele/ppp-subtypes

---

## License

MIT License. See `LICENSE` for details.
