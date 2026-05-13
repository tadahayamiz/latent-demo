# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Plot utilities for the CMap / estradiol Colab demo.

@author: tadahaya
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_compound_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str = "Compound-compound correlation",
    max_labels: int = 30,
):
    """Plot a compound-compound correlation heatmap."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    n = correlation.shape[0]
    ticks = np.arange(n) if n <= max_labels else np.linspace(0, n - 1, max_labels).astype(int)
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

    xy = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(compound_by_gene))
    plot_df = pd.DataFrame(xy, index=compound_by_gene.index, columns=["PC1", "PC2"]).join(clusters)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(plot_df["PC1"], plot_df["PC2"], c=plot_df["cluster"], s=30, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)

    for sample in plot_df.index:
        if any(term.lower() in sample.lower() for term in (highlight_terms or [])):
            ax.annotate(sample, (plot_df.loc[sample, "PC1"], plot_df.loc[sample, "PC2"]), fontsize=8)

    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="DBSCAN cluster")
    fig.tight_layout()
    return fig


def plot_dbscan_tsne_scatter(
    correlation: pd.DataFrame,
    clusters: pd.DataFrame,
    perplexity: float = 30.0,
    highlight_terms: list[str] | None = None,
    random_state: int = 0,
    title: str = "DBSCAN clusters on correlation distance",
):
    """Plot DBSCAN labels on a t-SNE map of correlation distance."""
    from sklearn.manifold import TSNE

    if not correlation.index.equals(clusters.index):
        raise ValueError("correlation.index and clusters.index must match.")
    if not 0 < float(perplexity) < correlation.shape[0]:
        raise ValueError("perplexity must be larger than 0 and smaller than the number of samples.")

    distance = np.clip(1.0 - correlation.to_numpy(), 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    xy = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=float(perplexity),
        learning_rate="auto",
        method="exact",
        max_iter=500,
        random_state=random_state,
    ).fit_transform(distance)

    plot_df = pd.DataFrame(xy, index=correlation.index, columns=["tSNE1", "tSNE2"]).join(clusters)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(plot_df["tSNE1"], plot_df["tSNE2"], c=plot_df["cluster"], s=30, alpha=0.85)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)

    for sample in plot_df.index:
        if any(term.lower() in sample.lower() for term in (highlight_terms or [])):
            ax.annotate(sample, (plot_df.loc[sample, "tSNE1"], plot_df.loc[sample, "tSNE2"]), fontsize=8)

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
    plot_df = pd.concat(
        [ranked_scores.head(top_n).assign(group="Top"), ranked_scores.tail(top_n).assign(group="Bottom")]
    ).sort_values(score_col, ascending=True)

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
    """Plot held-out likelihood and reconstruction error by component number."""
    fig, ax1 = plt.subplots(figsize=(7.0, 4.8))
    ax1.plot(summary["n_components"], summary["train_reconstruction_mse"], marker="o")
    ax1.set_xlabel("number of components")
    ax1.set_ylabel("train reconstruction MSE")

    ax2 = ax1.twinx()
    ax2.plot(summary["n_components"], summary["test_log_likelihood"], marker="s")
    ax2.set_ylabel("test log-likelihood")
    ax1.set_title(title)
    fig.tight_layout()
    return fig
