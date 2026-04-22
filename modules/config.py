 
from __future__ import annotations
 
import dataclasses
import json
import logging
import sys
from pathlib import Path
 

@dataclasses.dataclass
class PipelineConfig:
    # Data
    use_geo:                bool = False
    geo_id:                 str = "GSE152795"
    geo_cache_dir:          str = "geo_cache"
    synthetic_n_samples:    int = 40
    synthetic_n_genes:      int = 8_000
    synthetic_n_subtypes:   int = 3
    min_count_threshold:    int = 5
    min_samples_expressed:  float = 0.2
    
    # Preprocessing
    normalization:      str = "tmm"  # "tmm" or "quantile"
    top_var_genes:      int = 1_000
    variance_method:    str = "mad"  # "mad" or "var"
    
    # Dimensionality Reduction
    method:                 str = "sparse_pca"  # "pca" or "sparse_pca"
    n_components:           int = 20
    ledoit_wolf_shrinkage:  bool = True
    low_resource_mode:      bool  = True         # caps RAM, uses PCA fallback
    max_ram_mb:             int = 1_500
    
    # Clustering
    k_range:            list = dataclasses.field(default_factory=lambda: list(range(2, 7)))
    n_iterations:       int = 100
    subsample_rate:     float = 0.70
    bootstrap_ci:       bool = True
    bootstrap_n:        int = 50
    stability_threshold: float = 0.60
    
    # Biological Prioritization
    use_ppp_genesets:   bool = True
    geneset_weighting:  float = 2.0
    run_pathway_hints:  bool = True
    marker_top_n:       int   = 25
    
    # Compute
    profile_runtime:    bool = True
    profile_memory:     bool = True
    n_jobs:             int = 1                     # None = auto-detect, 1 = no parallelism
    random_seed:        int = 42
    low_resource_mode:  bool = True
    
    # Output
    out_dir:            str = "ppp_results"
    save_intermediate:  bool = True
    write_report:       bool = True
    dpi:                   int   = 150

    # Derived (set at runtime, not by user)
    out_path:  Path = dataclasses.field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        self.out_path = Path(self.out_dir)
        self.out_path.mkdir(exist_ok=True)
        
        
    @classmethod
    def from_json(cls, path: str) -> PipelineConfig:
        
        """
        Load config from a JSON file.  Nested dicts are flattened automatically,
        so you can group settings under keys like "data", "clustering", etc.
 
        Example JSON:
            {
              "data": { "synthetic_n_samples": 60, "use_geo": false },
              "clustering": { "k_range": [2, 3, 4], "n_iterations": 200 }
            }
        """
        
        with open(path) as f:
            raw = json.load(f)
        flat = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)

            # also accept flat top-level keys
        flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
        
        valid_field = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in flat.items() if k in valid_field})
    
    def to_json(self, path: str):
        """Serialise config to JSON (excluding derived fields)."""
        d = dataclasses.asdict(self)
        d.pop("out_path", None)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
            
    def summary(self) -> str:
        lines = ["PipelineConfig:", "-" * 40]
        for f in dataclasses.fields(self):
            if f.name == "out_path":
                continue
            lines.append(f"  {f.name:<28} = {getattr(self, f.name)}")
        return "\n".join(lines)
    
    def setup_logging(cfg: PipelineConfig) -> None:
        """Configure root logger to write to stdout and a log file."""
        handlers = [logging.StreamHandler(sys.stdout)]
        if cfg.out_path:
            handlers.append(
                logging.FileHandler(cfg.out_path / "pipeline.log", mode="w")
            )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
            force=True,
        )