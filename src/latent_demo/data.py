# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Data loading utilities for the CMap / estradiol Colab demo.

@author: tadahaya
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import pandas as pd


@dataclass(frozen=True)
class CMapData:
    """Container for reference expression signatures.

    ``gene_by_compound`` keeps the original table orientation.  
    ``compound_by_gene`` is the analysis matrix used in the notebook.
    """

    gene_by_compound: pd.DataFrame
    compound_by_gene: pd.DataFrame
    selected_genes: list[str]


def get_ref_cmap_path() -> str:
    """Return the packaged ``ref_cmap.csv`` path."""
    return str(resources.files("latent_demo").joinpath("datasets", "ref_cmap.csv"))


def load_ref_cmap(csv_path: str | None = None) -> pd.DataFrame:
    """Load the reference expression signature table.

    The returned DataFrame has genes as rows and compounds/samples as columns.
    """
    df = pd.read_csv(csv_path or get_ref_cmap_path(), index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(axis=0, how="any")


def prepare_cmap_data(
    csv_path: str | None = None,
    n_top_genes: int | None = 3000,
    variance_filter: bool = True,
) -> CMapData:
    """Load expression signatures and optionally retain high-variance genes."""
    df = load_ref_cmap(csv_path)
    if variance_filter and n_top_genes is not None and n_top_genes < df.shape[0]:
        selected = df.var(axis=1).sort_values(ascending=False).head(n_top_genes).index.tolist()
        df = df.loc[selected].copy()
    else:
        selected = df.index.tolist()
    return CMapData(gene_by_compound=df, compound_by_gene=df.T.copy(), selected_genes=selected)
