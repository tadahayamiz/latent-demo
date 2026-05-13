# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Model wrappers for PCA and factor analysis.

@author: tadahaya
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelName = Literal["pca", "factor_analysis"]


@dataclass
class LatentModelResult:
    model_name: str
    estimator: object
    pipeline: Pipeline
    scores: pd.DataFrame
    loadings: pd.DataFrame
    reconstructed: pd.DataFrame
    model_input: pd.DataFrame
    variable_ids: list[str]
    n_factors: int
    standardize: bool
    reconstruction_mse: float
    log_likelihood: float | None
    unique_noise: pd.Series | None = None


def fit_latent_model(
    observed: pd.DataFrame,
    variable_ids: list[str],
    model: ModelName = "factor_analysis",
    n_factors: int = 3,
    standardize: bool = True,
    random_state: int = 42,
) -> LatentModelResult:
    """Fit PCA or factor analysis to observed variables.

    Missing values are median-imputed before model fitting, so that the notebook
    can demonstrate a missing-data scenario without requiring students to handle
    preprocessing code.
    """
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if n_factors > len(variable_ids):
        raise ValueError("n_factors cannot exceed the number of variables")

    x_df = observed[variable_ids].copy()
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scaler", StandardScaler()))

    if model == "pca":
        estimator = PCA(n_components=n_factors, random_state=random_state)
    elif model == "factor_analysis":
        try:
            estimator = FactorAnalysis(n_components=n_factors, random_state=random_state, rotation="varimax")
        except TypeError:  # old scikit-learn
            estimator = FactorAnalysis(n_components=n_factors, random_state=random_state)
    else:
        raise ValueError("model must be either 'pca' or 'factor_analysis'")

    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    scores_arr = pipe.fit_transform(x_df)

    # Retrieve the transformed matrix used by the final estimator.
    x_model_arr = x_df
    for name, step in pipe.steps[:-1]:
        x_model_arr = step.transform(x_model_arr)
    x_model = pd.DataFrame(x_model_arr, columns=variable_ids)

    estimator = pipe.named_steps["model"]
    loadings = _extract_loadings(estimator, model, variable_ids)
    scores_arr, loadings = _orient_components(scores_arr, loadings)
    scores = pd.DataFrame(scores_arr, columns=[f"LV{k + 1}" for k in range(n_factors)], index=observed.index)

    x_hat_arr = _reconstruct(estimator, model, scores_arr)
    reconstructed = pd.DataFrame(x_hat_arr, columns=variable_ids, index=observed.index)
    model_input = pd.DataFrame(np.asarray(x_model), columns=variable_ids, index=observed.index)

    reconstruction_mse = float(np.nanmean((model_input.to_numpy() - reconstructed.to_numpy()) ** 2))
    try:
        log_likelihood = float(estimator.score(model_input.to_numpy()))
    except Exception:
        log_likelihood = None

    unique_noise = None
    if hasattr(estimator, "noise_variance_"):
        unique_noise = pd.Series(estimator.noise_variance_, index=variable_ids, name="estimated_unique_noise")

    return LatentModelResult(
        model_name=model,
        estimator=estimator,
        pipeline=pipe,
        scores=scores,
        loadings=loadings,
        reconstructed=reconstructed,
        model_input=model_input,
        variable_ids=variable_ids,
        n_factors=n_factors,
        standardize=standardize,
        reconstruction_mse=reconstruction_mse,
        log_likelihood=log_likelihood,
        unique_noise=unique_noise,
    )


def _extract_loadings(estimator: object, model: str, variable_ids: list[str]) -> pd.DataFrame:
    if model == "pca":
        # PCA components are unit directions. Multiplying by sqrt(eigenvalue)
        # gives a loading-like quantity for interpretation.
        arr = estimator.components_.T * np.sqrt(estimator.explained_variance_)
    else:
        arr = estimator.components_.T
    return pd.DataFrame(arr, index=variable_ids, columns=[f"LV{k + 1}" for k in range(arr.shape[1])])


def _orient_components(scores: np.ndarray, loadings: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    scores = np.asarray(scores).copy()
    loadings = loadings.copy()
    for k, col in enumerate(loadings.columns):
        idx = int(np.argmax(np.abs(loadings[col].to_numpy())))
        if loadings.iloc[idx, k] < 0:
            loadings.iloc[:, k] *= -1
            scores[:, k] *= -1
    return scores, loadings


def _reconstruct(estimator: object, model: str, scores: np.ndarray) -> np.ndarray:
    if model == "pca":
        return estimator.inverse_transform(scores)
    # FactorAnalysis has components_ with shape n_components x n_features.
    return scores @ estimator.components_ + estimator.mean_


# -----------------------------------------------------------------------------
# CMap expression-signature factor extraction
# -----------------------------------------------------------------------------

from dataclasses import field
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split


@dataclass
class CMapFactorResult:
    """Result of fast varimax factor extraction for compound signatures."""

    scores: pd.DataFrame
    loadings: pd.DataFrame
    selected_factor: str
    ranked_scores: pd.DataFrame
    target_score: float
    model: object
    scaler: object
    n_components: int
    n_top_genes: int | None
    explained_variance_ratio: np.ndarray = field(repr=False)


def _varimax(phi: np.ndarray, gamma: float = 1.0, max_iter: int = 30, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal varimax rotation.

    This is used instead of scikit-learn's rotated FactorAnalysis for speed in
    Colab. The extraction is PCA-based, followed by varimax rotation, which is a
    practical exploratory factor analysis style for teaching.
    """
    p, k = phi.shape
    rotation = np.eye(k)
    last_obj = 0.0
    for _ in range(max_iter):
        rotated = phi @ rotation
        u, singular_values, vh = np.linalg.svd(
            phi.T
            @ (
                rotated**3
                - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated))
            ),
            full_matrices=False,
        )
        rotation = u @ vh
        obj = singular_values.sum()
        if last_obj > 0 and obj / last_obj < 1.0 + tol:
            break
        last_obj = obj
    return phi @ rotation, rotation


def fit_cmap_varimax_factors(
    compound_by_gene: pd.DataFrame,
    n_components: int = 40,
    target_sample: str = "estradiol",
    n_top_genes: int | None = None,
    random_state: int = 0,
    select_mode: str = "max_abs",
) -> CMapFactorResult:
    """Fit fast varimax factors to compound x gene signatures.

    Parameters
    ----------
    compound_by_gene:
        Rows are compounds and columns are genes.
    n_components:
        Number of latent axes. For this dataset, 40 often makes estrogen-like
        and anti-estrogen signatures more visible than 10.
    target_sample:
        Compound/sample used to select the factor of interest.
    select_mode:
        ``"max_abs"`` selects the largest absolute target score and flips the
        sign so the target is positive. ``"max"`` selects the largest signed
        target score.
    """
    if target_sample not in compound_by_gene.index:
        candidates = [x for x in compound_by_gene.index if target_sample.lower() in x.lower()]
        raise ValueError(f"target_sample='{target_sample}' was not found. Partial matches: {candidates[:20]}")

    n_samples, n_features = compound_by_gene.shape
    max_components = min(n_samples - 1, n_features)
    if n_components > max_components:
        raise ValueError(f"n_components={n_components} exceeds max allowed value {max_components}.")

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


def compute_compound_correlation(compound_by_gene: pd.DataFrame) -> pd.DataFrame:
    """Compute compound-compound Pearson correlation from gene signatures."""
    x_scaled = StandardScaler().fit_transform(compound_by_gene)
    corr = np.corrcoef(x_scaled)
    return pd.DataFrame(corr, index=compound_by_gene.index, columns=compound_by_gene.index)


def cluster_compounds_dbscan(
    correlation: pd.DataFrame,
    eps: float = 0.6,
    min_samples: int = 3,
) -> pd.DataFrame:
    """Cluster compounds using DBSCAN on correlation distance."""
    distance = np.clip(1.0 - correlation.to_numpy(), 0.0, 2.0)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(distance)
    return pd.DataFrame({"sample": correlation.index, "cluster": labels}).set_index("sample")


def _fast_pca_average_log_likelihood(pca: PCA, x: np.ndarray) -> float:
    """Fast PPCA-like average log-likelihood for standardized test data.

    This avoids ``PCA.score`` because the latter may build a large precision
    matrix when the number of genes is high. The computation uses the PCA basis:
    selected PC directions have variances ``explained_variance_`` and the
    orthogonal residual subspace has variance ``noise_variance_``.
    """
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


def compare_cmap_factor_numbers(
    compound_by_gene: pd.DataFrame,
    component_grid: list[int] | tuple[int, ...] = (5, 10, 20, 40, 60),
    random_state: int = 0,
    test_size: float = 0.35,
) -> pd.DataFrame:
    """Compare latent dimension numbers by held-out PPCA/PCA log-likelihood.

    This is deliberately fast. It uses a PPCA-like likelihood computed in the
    PCA basis, so it remains usable for thousands of genes in Colab.
    """
    train, test = train_test_split(compound_by_gene, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train)
    x_test = scaler.transform(test)

    rows: list[dict[str, float | int]] = []
    max_components = min(x_train.shape[0] - 1, x_train.shape[1])
    for q in component_grid:
        q = int(q)
        if q < 1 or q > max_components:
            continue
        pca = PCA(n_components=q, svd_solver="randomized", random_state=random_state)
        train_scores = pca.fit_transform(x_train)
        x_hat = pca.inverse_transform(train_scores)
        train_mse = float(np.mean((x_train - x_hat) ** 2))
        rows.append(
            {
                "n_components": q,
                "train_reconstruction_mse": train_mse,
                "test_log_likelihood": _fast_pca_average_log_likelihood(pca, x_test),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )
    return pd.DataFrame(rows)
