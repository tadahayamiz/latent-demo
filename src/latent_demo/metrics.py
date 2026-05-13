# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Metrics used in the advanced part of the demo.

@author: tadahaya
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .models import fit_latent_model


def latent_alignment(scores: pd.DataFrame, latent: pd.DataFrame, factor_ids: list[str]) -> pd.DataFrame:
    """Absolute correlations between estimated latent scores and true factors."""
    values: list[dict[str, object]] = []
    for lv in scores.columns:
        for factor in factor_ids:
            corr = np.corrcoef(scores[lv].to_numpy(), latent[factor].to_numpy())[0, 1]
            values.append({"estimated_axis": lv, "true_factor": factor, "abs_correlation": abs(float(corr))})
    return pd.DataFrame(values).pivot(index="estimated_axis", columns="true_factor", values="abs_correlation")


def evaluate_factor_numbers(
    observed: pd.DataFrame,
    variable_ids: list[str],
    model: str = "factor_analysis",
    max_factors: int = 5,
    standardize: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare factor numbers by reconstruction error and test log-likelihood."""
    train_idx, test_idx = train_test_split(np.arange(len(observed)), test_size=0.35, random_state=random_state)
    train = observed.iloc[train_idx].reset_index(drop=True)
    test = observed.iloc[test_idx].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    upper = min(max_factors, len(variable_ids))
    for q in range(1, upper + 1):
        fit = fit_latent_model(train, variable_ids, model=model, n_factors=q, standardize=standardize, random_state=random_state)
        # Apply train preprocessing and estimator to test data.
        x_test_arr = test[variable_ids].copy()
        for name, step in fit.pipeline.steps[:-1]:
            x_test_arr = step.transform(x_test_arr)
        x_test = pd.DataFrame(x_test_arr, columns=variable_ids)
        estimator = fit.estimator
        test_ll = None
        try:
            test_ll = float(estimator.score(x_test.to_numpy()))
        except Exception:
            pass
        rows.append(
            {
                "n_factors": q,
                "train_reconstruction_mse": fit.reconstruction_mse,
                "test_log_likelihood": test_ll,
            }
        )
    return pd.DataFrame(rows)
