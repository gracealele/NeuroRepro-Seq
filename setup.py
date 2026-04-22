"""
setup.py
========
Install the ppp_subtypes package:

    # Editable install (recommended during development)
    pip install -e .

    # Standard install
    pip install .

    # With optional extras
    pip install -e ".[geo]"       # enables GEO download
    pip install -e ".[umap]"      # enables UMAP visualisation
    pip install -e ".[full]"      # all optional dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    # ── Identity ──────────────────────────────────────────────────────────────
    name             = "ppp-subtypes",
    version          = "2.0.0",
    author           = "Grace Alele",
    description      = (
        "Molecular disease subtype discovery pipeline for Postpartum Psychosis. "
        "HDLSS-aware, low-resource, PPP-biology-informed."
    ),
    long_description          = long_description,
    long_description_content_type = "text/markdown",
    license          = "MIT",
    python_requires  = ">=3.10",

    # ── Package discovery ─────────────────────────────────────────────────────
    packages = find_packages(exclude=["*.tests", "*.tests.*"]),
    include_package_data = True,

    # ── Core dependencies ─────────────────────────────────────────────────────
    install_requires = [
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "scipy>=1.10",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "psutil>=5.9",
    ],

    # ── Optional extras ───────────────────────────────────────────────────────
    extras_require = {
        "geo": [
            "GEOparse>=2.0",          # live GEO series download
        ],
        "umap": [
            "umap-learn>=0.5",        # UMAP 2-D visualisation
        ],
        "full": [
            "GEOparse>=2.0",
            "umap-learn>=0.5",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },

    # ── CLI entry point ───────────────────────────────────────────────────────
    entry_points = {
        "console_scripts": [
            # After install: run `ppp-subtypes` from anywhere
            "ppp-subtypes = ppp_subtypes.main:main",
        ],
    },

    # ── PyPI classifiers ──────────────────────────────────────────────────────
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],

    keywords = [
        "postpartum psychosis", "transcriptomics", "RNA-seq",
        "molecular subtypes", "consensus clustering", "HDLSS",
        "bioinformatics", "reproductive health", "low-resource",
    ],
)
