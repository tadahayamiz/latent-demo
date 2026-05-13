# latent-demo

Small Colab demos for latent variables, latent states, and latent representations.

This repository currently contains a minimal teaching demo for a pharmaceutical informatics lecture:

> **観測データから見えない個体状態を推定する：PCA・因子分析による潜在変数モデリング**

The repository is intentionally small. The notebook is the student-facing entry point, while reusable code lives under `src/latent_demo`.

## Student entry point

Open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizuno-group/latent-demo/blob/v0.1.0/notebooks/01_pharm_info_latent_toxicity.ipynb)

Recommended instruction for students:

1. Open the Colab link above.
2. Click **Copy to Drive**.
3. Run the cells from top to bottom.
4. Change only the form inputs unless instructed otherwise.

## Colab installation

The notebook installs the package from a tagged GitHub zip:

```python
%pip install -q https://github.com/mizuno-group/latent-demo/archive/refs/tags/v0.1.0.zip
```

This does not require a student GitHub account when the repository is public.

For development, you can also install from the repository root:

```bash
pip install -e .
```

## Repository structure

```text
latent-demo/
├── pyproject.toml
├── README.md
├── notebooks/
│   └── 01_pharm_info_latent_toxicity.ipynb
├── src/
│   └── latent_demo/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       ├── metrics.py
│       ├── plots.py
│       ├── layout.py
│       ├── workflow.py
│       └── scenarios/
│           ├── pharm_toxicity_basic.yaml
│           ├── pharm_toxicity_noisy.yaml
│           └── pharm_toxicity_missing.yaml
└── tests/
    └── test_smoke.py
```

## Demo concept

The demo creates synthetic clinical-lab-like data from three hidden latent states:

- 肝障害状態
- 炎症状態
- 骨髄抑制状態

Students initially see only observed variables such as ALT, AST, CRP, WBC, PLT, and Hb. They then compare PCA and factor analysis and interpret latent axes from the loading matrix.

The teaching focus is not coding. Students mainly:

- run cells,
- switch model/scenario settings,
- inspect figures and loading tables,
- name latent axes,
- compare results with and without standardization,
- optionally reveal the true hidden factors at the end.

## Scenarios

| Scenario | Purpose |
|---|---|
| `pharm_toxicity_basic` | Clean synthetic data where latent factors are relatively easy to recover |
| `pharm_toxicity_noisy` | Higher variable-specific noise; useful for comparing PCA and factor analysis |
| `pharm_toxicity_missing` | Includes missing values; useful for discussing preprocessing and imputation |

## Instructor notes

The notebook should be treated as a stable lecture handout. Development should be done in `src/latent_demo`, and the notebook should remain thin.

Recommended release workflow:

1. Confirm that tests pass.
2. Push to GitHub under `mizuno-group/latent-demo`.
3. Create a release tag such as `v0.1.0`.
4. Use the tagged Colab URL in slides or LMS.

## Test

```bash
pytest -q
```

## Authors
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- tadahaya[at]gmail.com  
