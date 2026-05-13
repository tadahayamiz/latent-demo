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
