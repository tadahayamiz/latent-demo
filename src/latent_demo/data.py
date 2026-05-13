# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Synthetic pharmaceutical data generation for latent variable demos.

@author: tadahaya
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratedData:
    """Container for synthetic demo data."""

    observed: pd.DataFrame
    latent: pd.DataFrame
    loadings: pd.DataFrame
    unique_noise: pd.Series
    labels: pd.DataFrame
    variable_ids: list[str]
    factor_ids: list[str]


def generate_synthetic_data(cfg: dict[str, Any]) -> GeneratedData:
    """Generate observed variables from hidden latent factors.

    The generative model is

        X_raw = baseline + scale * (Z @ Lambda.T + noise)

    where Z is hidden from students until ``show_truth=True`` in the notebook.
    """
    dataset_cfg = cfg["dataset"]
    rng = np.random.default_rng(int(dataset_cfg.get("seed", 42)))
    n_samples = int(dataset_cfg.get("n_samples", 240))
    global_noise = float(dataset_cfg.get("noise_level", 1.0))
    missing_rate = float(dataset_cfg.get("missing_rate", 0.0))

    factor_ids = [item["id"] for item in cfg["latent_factors"]]
    factor_labels = {item["id"]: item.get("label_ja", item["id"]) for item in cfg["latent_factors"]}
    q = len(factor_ids)

    corr = np.asarray(dataset_cfg.get("latent_correlation", np.eye(q)), dtype=float)
    if corr.shape != (q, q):
        raise ValueError("dataset.latent_correlation must be q x q where q is the number of factors.")
    z = rng.multivariate_normal(mean=np.zeros(q), cov=corr, size=n_samples)

    variables = cfg["variables"]
    variable_ids = [item["id"] for item in variables]
    variable_labels = {item["id"]: item.get("label_ja", item["id"]) for item in variables}

    loadings = np.zeros((len(variable_ids), q), dtype=float)
    unique_noise = np.zeros(len(variable_ids), dtype=float)
    baselines = np.zeros(len(variable_ids), dtype=float)
    scales = np.ones(len(variable_ids), dtype=float)

    for j, variable in enumerate(variables):
        for k, factor_id in enumerate(factor_ids):
            loadings[j, k] = float(variable.get("loadings", {}).get(factor_id, 0.0))
        unique_noise[j] = float(variable.get("unique_noise", 0.5))
        baselines[j] = float(variable.get("baseline", 0.0))
        scales[j] = float(variable.get("scale", 1.0))

    structured_signal = z @ loadings.T
    noise = rng.normal(loc=0.0, scale=global_noise * unique_noise, size=structured_signal.shape)
    x_standardized = structured_signal + noise
    x_raw = baselines + scales * x_standardized

    observed = pd.DataFrame(x_raw, columns=variable_ids)
    observed.insert(0, "sample_id", [f"S{i + 1:03d}" for i in range(n_samples)])

    if missing_rate > 0:
        mask = rng.uniform(size=(n_samples, len(variable_ids))) < missing_rate
        observed.loc[:, variable_ids] = observed.loc[:, variable_ids].mask(mask)

    latent = pd.DataFrame(z, columns=factor_ids)
    latent.insert(0, "sample_id", observed["sample_id"])

    dominant_factor = np.asarray([factor_ids[i] for i in np.argmax(z, axis=1)])
    high_risk = ((z[:, factor_ids.index("liver_injury")] if "liver_injury" in factor_ids else z[:, 0]) > 0.8).astype(int)
    labels = pd.DataFrame(
        {
            "sample_id": observed["sample_id"],
            "dominant_latent_factor": dominant_factor,
            "dominant_latent_factor_ja": [factor_labels[x] for x in dominant_factor],
            "high_liver_risk": high_risk,
        }
    )

    loadings_df = pd.DataFrame(loadings, index=variable_ids, columns=factor_ids)
    loadings_df.index.name = "variable"
    loadings_df.attrs["factor_labels"] = factor_labels
    loadings_df.attrs["variable_labels"] = variable_labels

    unique_noise_s = pd.Series(unique_noise, index=variable_ids, name="unique_noise")
    return GeneratedData(
        observed=observed,
        latent=latent,
        loadings=loadings_df,
        unique_noise=unique_noise_s,
        labels=labels,
        variable_ids=variable_ids,
        factor_ids=factor_ids,
    )


# -----------------------------------------------------------------------------
# CMap expression-signature demo utilities
# -----------------------------------------------------------------------------

from importlib import resources


@dataclass(frozen=True)
class CMapData:
    """Container for CMap-like reference expression signatures.

    Attributes
    ----------
    gene_by_compound:
        Original matrix with genes as rows and compounds as columns.
    compound_by_gene:
        Transposed matrix used for analysis: compounds as rows and genes as columns.
    selected_genes:
        Genes retained after variance filtering.
    """

    gene_by_compound: pd.DataFrame
    compound_by_gene: pd.DataFrame
    selected_genes: list[str]


def get_ref_cmap_path() -> str:
    """Return the packaged ``ref_cmap.csv`` path.

    The CSV is intentionally bundled inside the package so that students do not
    need to upload any data file during the Colab exercise.
    """
    ref = resources.files("latent_demo").joinpath("datasets", "ref_cmap.csv")
    return str(ref)


def load_ref_cmap(csv_path: str | None = None) -> pd.DataFrame:
    """Load the reference CMap-like expression signature table.

    The returned DataFrame has genes as rows and compounds/samples as columns.
    """
    path = csv_path or get_ref_cmap_path()
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")
    return df


def prepare_cmap_data(
    csv_path: str | None = None,
    n_top_genes: int | None = 3000,
    variance_filter: bool = True,
) -> CMapData:
    """Load and optionally variance-filter expression signatures.

    Parameters
    ----------
    csv_path:
        Optional external CSV path. If omitted, the packaged dataset is used.
    n_top_genes:
        Number of high-variance genes to retain. ``None`` keeps all genes.
    variance_filter:
        If true, retain the genes with the largest variance across compounds.
    """
    df = load_ref_cmap(csv_path)
    if variance_filter and n_top_genes is not None and n_top_genes < df.shape[0]:
        selected = df.var(axis=1).sort_values(ascending=False).head(n_top_genes).index.tolist()
        df_use = df.loc[selected].copy()
    else:
        selected = df.index.tolist()
        df_use = df.copy()
    return CMapData(gene_by_compound=df_use, compound_by_gene=df_use.T.copy(), selected_genes=selected)
