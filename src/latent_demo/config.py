# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Scenario loading utilities.
A scenario is a small YAML file describing the latent factors, observed
variables, and noise levels used in a teaching demo.

@author: tadahaya
"""
from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_PACKAGE = "latent_demo"
_SCENARIO_DIR = "scenarios"


def list_scenarios() -> list[str]:
    """Return available built-in scenario IDs without file extensions."""
    scenario_dir = resources.files(_PACKAGE).joinpath(_SCENARIO_DIR)
    return sorted(path.stem for path in scenario_dir.iterdir() if path.name.endswith(".yaml"))


def load_scenario(name: str | Path) -> dict[str, Any]:
    """Load a scenario YAML file.

    Parameters
    ----------
    name:
        Either a built-in scenario ID such as ``"pharm_toxicity_basic"``
        or a path to a YAML file.
    """
    path = Path(name)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        if path.suffix:
            scenario_name = path.name
        else:
            scenario_name = f"{name}.yaml"
        scenario_path = resources.files(_PACKAGE).joinpath(_SCENARIO_DIR, scenario_name)
        if not scenario_path.is_file():
            available = ", ".join(list_scenarios())
            raise FileNotFoundError(f"Scenario '{name}' was not found. Available: {available}")
        text = scenario_path.read_text(encoding="utf-8")

    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError("Scenario YAML must define a mapping at the top level.")
    _validate_scenario(cfg)
    return deepcopy(cfg)


def _validate_scenario(cfg: dict[str, Any]) -> None:
    required = ["scenario", "dataset", "latent_factors", "variables"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Scenario is missing required keys: {missing}")

    factor_ids = [factor["id"] for factor in cfg["latent_factors"]]
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("Latent factor IDs must be unique.")

    for variable in cfg["variables"]:
        if "id" not in variable:
            raise ValueError("Every variable must have an 'id'.")
        loadings = variable.get("loadings", {})
        unknown = set(loadings) - set(factor_ids)
        if unknown:
            raise ValueError(f"Variable {variable['id']} has unknown loading keys: {sorted(unknown)}")
