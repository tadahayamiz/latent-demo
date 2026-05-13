# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Plotting helpers for the latent variable demo.
The functions intentionally use plain matplotlib so that the notebook runs on a
fresh Colab runtime with minimal dependencies.

@author: tadahaya
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_correlation_heatmap(observed: pd.DataFrame, variable_ids: list[str], title: str = "Correlation among observed variables"):
    corr = observed[variable_ids].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(variable_ids)))
    ax.set_yticks(np.arange(len(variable_ids)))
    ax.set_xticklabels(variable_ids, rotation=45, ha="right")
    ax.set_yticklabels(variable_ids)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_scores(scores: pd.DataFrame, labels: pd.DataFrame | None = None, show_truth: bool = False, title: str = "Estimated latent space"):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if scores.shape[1] == 1:
        ax.scatter(scores.iloc[:, 0], np.zeros(len(scores)), s=28, alpha=0.8)
        ax.set_ylabel("0")
    elif show_truth and labels is not None and "high_liver_risk" in labels:
        ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], c=labels["high_liver_risk"], s=28, alpha=0.8)
    else:
        ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], s=28, alpha=0.8)
    ax.set_xlabel(scores.columns[0])
    ax.set_ylabel(scores.columns[1] if scores.shape[1] > 1 else "")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_loadings_heatmap(loadings: pd.DataFrame, title: str = "Loadings"):
    fig_height = max(4.0, 0.35 * len(loadings.index) + 1.5)
    fig, ax = plt.subplots(figsize=(5.8, fig_height))
    vmax = np.nanmax(np.abs(loadings.to_numpy()))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(loadings.to_numpy(), vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(loadings.shape[1]))
    ax.set_yticks(np.arange(loadings.shape[0]))
    ax.set_xticklabels(loadings.columns)
    ax.set_yticklabels(loadings.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_unique_noise(unique_noise: pd.Series, title: str = "Estimated variable-specific noise"):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    unique_noise.plot(kind="bar", ax=ax)
    ax.set_ylabel("estimated noise variance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_alignment(alignment: pd.DataFrame, title: str = "Alignment with hidden true factors"):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(alignment.to_numpy(), vmin=0, vmax=1)
    ax.set_xticks(np.arange(alignment.shape[1]))
    ax.set_yticks(np.arange(alignment.shape[0]))
    ax.set_xticklabels(alignment.columns, rotation=45, ha="right")
    ax.set_yticklabels(alignment.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_model_selection(summary: pd.DataFrame, title: str = "Model comparison by number of factors"):
    fig, ax1 = plt.subplots(figsize=(6.8, 4.6))
    ax1.plot(summary["n_factors"], summary["train_reconstruction_mse"], marker="o", label="train reconstruction MSE")
    ax1.set_xlabel("number of latent factors")
    ax1.set_ylabel("train reconstruction MSE")
    if "test_log_likelihood" in summary and summary["test_log_likelihood"].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(summary["n_factors"], summary["test_log_likelihood"], marker="s", label="test log-likelihood")
        ax2.set_ylabel("test log-likelihood")
    ax1.set_title(title)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# CMap expression-signature demo plots
# -----------------------------------------------------------------------------


def plot_compound_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str = "Compound-compound correlation",
    max_labels: int = 30,
):
    """Plot a compound-compound correlation heatmap.

    For readability, only a subset of tick labels is shown when many compounds
    are present.
    """
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    n = correlation.shape[0]
    if n <= max_labels:
        ticks = np.arange(n)
    else:
        ticks = np.linspace(0, n - 1, max_labels).astype(int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(correlation.columns[ticks], rotation=90, fontsize=7)
    ax.set_yticklabels(correlation.index[ticks], fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_dbscan_pca_scatter(
    compound_by_gene: pd.DataFrame,
    clusters: pd.DataFrame,
    highlight_terms: list[str] | None = None,
    title: str = "DBSCAN clusters on compound signatures",
):
    """Plot a 2D PCA view colored by DBSCAN cluster labels."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x_scaled = StandardScaler().fit_transform(compound_by_gene)
    xy = PCA(n_components=2, random_state=0).fit_transform(x_scaled)
    plot_df = pd.DataFrame(xy, index=compound_by_gene.index, columns=["PC1", "PC2"]).join(clusters)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(plot_df["PC1"], plot_df["PC2"], c=plot_df["cluster"], s=30, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)

    highlight_terms = highlight_terms or []
    for sample in plot_df.index:
        if any(term.lower() in sample.lower() for term in highlight_terms):
            ax.annotate(sample, (plot_df.loc[sample, "PC1"], plot_df.loc[sample, "PC2"]), fontsize=8)

    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="DBSCAN cluster")
    fig.tight_layout()
    return fig


def plot_ranked_factor_scores(
    ranked_scores: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top and bottom samples on selected factor",
):
    """Plot top and bottom compounds by selected factor score."""
    score_col = "factor_score"
    top = ranked_scores.head(top_n).copy()
    bottom = ranked_scores.tail(top_n).sort_values(score_col, ascending=True).copy()
    plot_df = pd.concat([top.assign(group="Top"), bottom.assign(group="Bottom")])
    plot_df = plot_df.sort_values(score_col, ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.26 * len(plot_df))))
    ax.barh(plot_df.index, plot_df[score_col])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("factor score")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cmap_factor_number_summary(
    summary: pd.DataFrame,
    title: str = "Factor number comparison",
):
    """Plot held-out likelihood and reconstruction error by factor number."""
    fig, ax1 = plt.subplots(figsize=(7.0, 4.8))
    ax1.plot(summary["n_components"], summary["train_reconstruction_mse"], marker="o", label="train reconstruction MSE")
    ax1.set_xlabel("number of components")
    ax1.set_ylabel("train reconstruction MSE")
    ax2 = ax1.twinx()
    ax2.plot(summary["n_components"], summary["test_log_likelihood"], marker="s", label="test log-likelihood")
    ax2.set_ylabel("test log-likelihood")
    ax1.set_title(title)
    fig.tight_layout()
    return fig
