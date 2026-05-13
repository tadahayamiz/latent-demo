# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Small Colab demos for latent variables and compound expression signatures.

@author: tadahaya
"""

from .cmap import (
    ANTI_ESTROGEN_TERMS,
    ESTROGEN_TERMS,
    CMapData,
    CMapFactorResult,
    cluster_compounds_dbscan,
    compare_cmap_factor_numbers,
    compute_compound_correlation,
    fit_cmap_varimax_factors,
    load_ref_cmap,
    rank_target_correlations,
    match_samples,
    plot_cmap_factor_number_summary,
    plot_compound_correlation_heatmap,
    plot_dbscan_pca_scatter,
    plot_dbscan_tsne_scatter,
    plot_ranked_factor_scores,
    prepare_cmap_data,
    summarize_terms,
)

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

__version__ = "0.2.2"
