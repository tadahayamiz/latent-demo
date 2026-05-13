# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Model utilities for the CMap / estradiol Colab demo.

@author: tadahaya
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class CMapFactorResult:
    """Result of PCA-based varimax factor extraction."""

    scores: pd.DataFrame
    loadings: pd.DataFrame
    selected_factor: str
    ranked_scores: pd.DataFrame
    target_score: float
    model: PCA
    scaler: StandardScaler
    n_components: int
    n_top_genes: int | None
    explained_variance_ratio: np.ndarray = field(repr=False)


def compute_compound_correlation(compound_by_gene: pd.DataFrame) -> pd.DataFrame:
    """Compute compound-compound Pearson correlation from gene signatures."""
    x_scaled = StandardScaler().fit_transform(compound_by_gene)
    corr = np.corrcoef(x_scaled)
    return pd.DataFrame(corr, index=compound_by_gene.index, columns=compound_by_gene.index)


def rank_target_correlations(
    correlation: pd.DataFrame,
    target_sample: str,
) -> pd.DataFrame:
    """Rank compounds by correlation to a target compound."""
    if target_sample not in correlation.columns:
        candidates = [x for x in correlation.columns if target_sample.lower() in x.lower()]
        raise ValueError(f"target_sample='{target_sample}' was not found. Partial matches: {candidates[:20]}")
    ranked = correlation[target_sample].drop(index=target_sample).sort_values(ascending=False).to_frame("correlation")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def cluster_compounds_dbscan(
    correlation: pd.DataFrame,
    eps: float = 0.6,
    min_samples: int = 3,
) -> pd.DataFrame:
    """Cluster compounds by DBSCAN using correlation distance."""
    distance = np.clip(1.0 - correlation.to_numpy(), 0.0, 2.0)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(distance)
    return pd.DataFrame({"sample": correlation.index, "cluster": labels}).set_index("sample")


def fit_cmap_varimax_factors(
    compound_by_gene: pd.DataFrame,
    n_components: int = 40,
    target_sample: str = "estradiol",
    n_top_genes: int | None = None,
    random_state: int = 0,
    select_mode: str = "max_abs",
) -> CMapFactorResult:
    """Fit PCA-based varimax factors and select the target-associated factor."""
    if target_sample not in compound_by_gene.index:
        candidates = [x for x in compound_by_gene.index if target_sample.lower() in x.lower()]
        raise ValueError(f"target_sample='{target_sample}' was not found. Partial matches: {candidates[:20]}")

    n_samples, n_features = compound_by_gene.shape
    max_components = min(n_samples - 1, n_features)
    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components={n_components} must be between 1 and {max_components}.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(compound_by_gene)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    raw_scores = pca.fit_transform(x_scaled)
    raw_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    rotated_loadings, rotation = _varimax(raw_loadings)
    rotated_scores = raw_scores @ rotation

    score_df = pd.DataFrame(
        rotated_scores,
        index=compound_by_gene.index,
        columns=[f"Factor{i + 1}" for i in range(n_components)],
    )
    loading_df = pd.DataFrame(rotated_loadings, index=compound_by_gene.columns, columns=score_df.columns)

    if select_mode == "max_abs":
        selected_factor = score_df.loc[target_sample].abs().idxmax()
        if score_df.loc[target_sample, selected_factor] < 0:
            score_df[selected_factor] *= -1
            loading_df[selected_factor] *= -1
    elif select_mode == "max":
        selected_factor = score_df.loc[target_sample].idxmax()
    else:
        raise ValueError("select_mode must be 'max_abs' or 'max'.")

    ranked = score_df[[selected_factor]].sort_values(selected_factor, ascending=False)
    ranked = ranked.rename(columns={selected_factor: "factor_score"})
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    return CMapFactorResult(
        scores=score_df,
        loadings=loading_df,
        selected_factor=selected_factor,
        ranked_scores=ranked,
        target_score=float(score_df.loc[target_sample, selected_factor]),
        model=pca,
        scaler=scaler,
        n_components=n_components,
        n_top_genes=n_top_genes,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def compare_cmap_factor_numbers(
    compound_by_gene: pd.DataFrame,
    component_grid: list[int] | tuple[int, ...] = (5, 10, 20, 40, 60),
    random_state: int = 0,
    test_size: float = 0.35,
) -> pd.DataFrame:
    """Compare component numbers by reconstruction error and held-out PPCA-like likelihood."""
    train, test = train_test_split(compound_by_gene, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train)
    x_test = scaler.transform(test)

    rows: list[dict[str, float | int]] = []
    max_components = min(x_train.shape[0] - 1, x_train.shape[1])
    for q in component_grid:
        q = int(q)
        if not 1 <= q <= max_components:
            continue
        pca = PCA(n_components=q, svd_solver="randomized", random_state=random_state)
        train_scores = pca.fit_transform(x_train)
        x_hat = pca.inverse_transform(train_scores)
        rows.append(
            {
                "n_components": q,
                "train_reconstruction_mse": float(np.mean((x_train - x_hat) ** 2)),
                "test_log_likelihood": _fast_pca_average_log_likelihood(pca, x_test),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )
    return pd.DataFrame(rows)


def _varimax(phi: np.ndarray, gamma: float = 1.0, max_iter: int = 30, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal varimax rotation."""
    p, k = phi.shape
    rotation = np.eye(k)
    last_obj = 0.0
    for _ in range(max_iter):
        rotated = phi @ rotation
        u, singular_values, vh = np.linalg.svd(
            phi.T @ (rotated**3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated))),
            full_matrices=False,
        )
        rotation = u @ vh
        obj = singular_values.sum()
        if last_obj > 0 and obj / last_obj < 1.0 + tol:
            break
        last_obj = obj
    return phi @ rotation, rotation


def _fast_pca_average_log_likelihood(pca: PCA, x: np.ndarray) -> float:
    """Fast PPCA-like average log-likelihood for standardized test data."""
    n_samples, n_features = x.shape
    q = pca.components_.shape[0]
    centered = x - pca.mean_
    projected = centered @ pca.components_.T
    total_norm2 = np.sum(centered**2, axis=1)
    projected_norm2 = np.sum(projected**2, axis=1)
    residual_norm2 = np.maximum(total_norm2 - projected_norm2, 0.0)

    variances = np.maximum(pca.explained_variance_, 1e-8)
    noise_variance = float(max(getattr(pca, "noise_variance_", 0.0), 1e-8))
    log_det = float(np.sum(np.log(variances)) + (n_features - q) * np.log(noise_variance))
    quad = np.sum((projected**2) / variances, axis=1) + residual_norm2 / noise_variance
    ll = -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + quad)
    return float(np.mean(ll))
