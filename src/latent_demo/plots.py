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
