# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Small Colab demos for latent variables and latent representations.

@author: tadahaya
"""

from .config import load_scenario, list_scenarios
from .data import load_ref_cmap, prepare_cmap_data
from .models import (
    cluster_compounds_dbscan,
    compare_cmap_factor_numbers,
    compute_compound_correlation,
    fit_cmap_varimax_factors,
)
from .workflow import (
    compare_cmap_factor_numbers_workflow,
    run_cmap_estradiol_demo,
    run_latent_toxicity_demo,
    compare_factor_numbers,
)

__all__ = [
    "load_scenario",
    "list_scenarios",
    "load_ref_cmap",
    "prepare_cmap_data",
    "compute_compound_correlation",
    "cluster_compounds_dbscan",
    "fit_cmap_varimax_factors",
    "compare_cmap_factor_numbers",
    "run_cmap_estradiol_demo",
    "compare_cmap_factor_numbers_workflow",
    "run_latent_toxicity_demo",
    "compare_factor_numbers",
]

__version__ = "0.2.1"
