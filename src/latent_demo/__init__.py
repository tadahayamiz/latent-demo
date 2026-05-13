# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Small Colab demos for latent variables and latent representations.

@author: tadahaya
"""

from .config import load_scenario, list_scenarios
from .workflow import run_latent_toxicity_demo, compare_factor_numbers

__all__ = [
    "load_scenario",
    "list_scenarios",
    "run_latent_toxicity_demo",
    "compare_factor_numbers",
]

__version__ = "0.1.0"
