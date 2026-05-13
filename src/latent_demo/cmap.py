# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Stepwise public API for the CMap / estradiol Colab demo.

@author: tadahaya
"""
from __future__ import annotations

import pandas as pd

from .data import CMapData, load_ref_cmap, prepare_cmap_data
from .models import (
    CMapFactorResult,
    cluster_compounds_dbscan,
    compare_cmap_factor_numbers,
    compute_compound_correlation,
    fit_cmap_varimax_factors,
    rank_target_correlations,
)
from .plots import (
    plot_cmap_factor_number_summary,
    plot_compound_correlation_heatmap,
    plot_dbscan_pca_scatter,
    plot_dbscan_tsne_scatter,
    plot_ranked_factor_scores,
)

ESTROGEN_TERMS = [
    "estradiol",
    "estrone",
    "estriol",
    "estropipate",
    "diethylstilbestrol",
    "dienestrol",
    "equilin",
    "prasterone",
]

ANTI_ESTROGEN_TERMS = [
    "tamoxifen",
    "raloxifene",
    "fulvestrant",
    "clomifene",
    "mifepristone",
]


def match_samples(index, terms: list[str]) -> list[str]:
    """Return sample names that contain any keyword in ``terms``."""
    return [x for x in index if any(term.lower() in x.lower() for term in terms)]


def summarize_terms(
    scores: pd.DataFrame,
    estrogen_terms: list[str] | None = None,
    anti_estrogen_terms: list[str] | None = None,
) -> pd.DataFrame:
    """Extract estrogen-like and anti-estrogen rows from a factor score table."""
    rows = []
    for name in match_samples(scores.index, estrogen_terms or ESTROGEN_TERMS):
        rows.append((name, "estrogen-like"))
    for name in match_samples(scores.index, anti_estrogen_terms or ANTI_ESTROGEN_TERMS):
        rows.append((name, "anti-estrogen"))
    if not rows:
        return pd.DataFrame(columns=list(scores.columns) + ["category"])

    out = scores.loc[[name for name, _ in rows]].copy()
    out["category"] = [category for _, category in rows]
    return out


__all__ = [
    "CMapData",
    "CMapFactorResult",
    "load_ref_cmap",
    "prepare_cmap_data",
    "compute_compound_correlation",
    "rank_target_correlations",
    "cluster_compounds_dbscan",
    "fit_cmap_varimax_factors",
    "compare_cmap_factor_numbers",
    "plot_compound_correlation_heatmap",
    "plot_dbscan_pca_scatter",
    "plot_dbscan_tsne_scatter",
    "plot_ranked_factor_scores",
    "plot_cmap_factor_number_summary",
    "ESTROGEN_TERMS",
    "ANTI_ESTROGEN_TERMS",
    "match_samples",
    "summarize_terms",
]
