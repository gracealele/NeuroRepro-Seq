"""
ppp_subtypes/modules/profiler.py
================================
Lightweight context-manager profiler tracking:
  • wall-clock time per pipeline stage
  • peak RAM per stage (via tracemalloc)
 
Usage
-----
    from ppp_subtypes.modules.profiler import Profiler
 
    prof = Profiler()
    with prof("preprocessing"):
        ...   # code to time
 
    df = prof.summary()   # pandas DataFrame
    prof.save(out_dir)    # saves CSV + bar chart
"""

from __future__ import annotations


import logging
import time
import tracemalloc

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ===============================================================
# PROFILE DECORATOR
# ===============================================================

class Profiler:
    """
    Context-manager profiler.  Each named block is timed and its peak
    memory usage is recorded.  Designed to be low-overhead — tracemalloc
    is only active inside a block, not across the whole pipeline.
    """


    def __init__(self, active: bool = True):
        self.active = active
        self._log: list[dict] = []
        
    def __call__(self, stage: str) -> "_Stage":
        return _Stage(self, stage)
    
    def summary(self) -> pd.DataFrame:
        """Return all recorded stages as a DataFrame."""
        return pd.DataFrame(self._log)
    
    def save(self, out_dir: Path, dpi: int = 150) -> None:
        """Write profiler_log.csv and profiler_timing.png to out_dir."""
        df = self.summary()
        if df.empty:
            return
 
        csv_path = out_dir / "profiler_log.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"[Profiler] Log → {csv_path}")
 
        # Bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, len(df) * 0.4)))
        colours = [f"C{i}" for i in range(len(df))]
 
        ax1.barh(df["stage"], df["time_s"], color=colours)
        ax1.set_xlabel("Wall-clock time (s)")
        ax1.set_title("Runtime per Stage", fontweight="bold")
        for spine in ["top", "right"]:
            ax1.spines[spine].set_visible(False)
 
        ax2.barh(df["stage"], df["peak_ram_mb"], color=colours)
        ax2.set_xlabel("Peak RAM (MB)")
        ax2.set_title("Memory per Stage", fontweight="bold")
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
 
        total_t = df["time_s"].sum()
        peak_m  = df["peak_ram_mb"].max()
        fig.suptitle(
            f"Compute Profile  —  total {total_t:.1f}s  |  peak {peak_m:.0f} MB",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        img_path = out_dir / "profiler_timing.png"
        fig.savefig(img_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[Profiler] Chart → {img_path}")
        logging.info(f"[Profiler] Total: {total_t:.1f}s | Peak RAM: {peak_m:.0f} MB")
    
    
    class _Stage:
        """Internal context manager for a single timed stage."""

        def __init__(self, parent: Profiler, name: str):
            self._parent = parent
            self._name = name
            self._t0: float = 0.0
            
        def __enter__(self) -> "_Stage":
            self._t0 = time.perf_counter()
            if self._parent.active:
                tracemalloc.start()
            return self
                
        def __exit__(self, *_) -> None:
            elapsed = time.perf_counter() - self.t0
            peak_mb = 0.0
            if self._parent.active:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = peak / 1e6
            
            self._parent._log.append({
                "stage":        self._name,
                "time_s":       round(elapsed, 3),
                "peak_ram_mb":  round(peak_mb, 1)
            })
            logging.info(f" ⏱ {self._name:<35}: {elapsed:6.2f}s, | RAM peak: {peak_mb:6.1f}MB")  
    
    def summmary(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)