# latent-demo

Small Colab demos for latent variables, latent states, and latent representations.

This repository currently contains a teaching demo for a pharmaceutical informatics lecture:

> **CMap発現シグネチャから estradiol-like factor を探す：相関・DBSCAN・varimax因子抽出**

The repository is intentionally small. The notebook is the student-facing entry point, while reusable code lives under `src/latent_demo`.

## Student entry point

Open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizuno-group/latent-demo/blob/v0.2.1/notebooks/01_cmap_estradiol_factor_demo.ipynb)

Recommended instruction for students:

1. Open the Colab link above.
2. Click **Copy to Drive**.
3. Run the cells from top to bottom.
4. Change only the form inputs unless instructed otherwise.

## Colab installation

The notebook installs the package from a tagged GitHub zip:

```python
%pip install -q https://github.com/mizuno-group/latent-demo/archive/refs/tags/v0.2.1.zip
```

The reference data `ref_cmap.csv` is bundled inside the package, so students do not need to upload a data file during the standard lecture workflow.

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
│   └── 01_cmap_estradiol_factor_demo.ipynb
├── src/
│   └── latent_demo/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── cmap.py
│       ├── models.py
│       ├── metrics.py
│       ├── plots.py
│       ├── layout.py
│       ├── workflow.py
│       ├── datasets/
│       │   └── ref_cmap.csv
│       └── scenarios/
│           ├── pharm_toxicity_basic.yaml
│           ├── pharm_toxicity_noisy.yaml
│           └── pharm_toxicity_missing.yaml
└── tests/
    └── test_smoke.py
```

## Demo concept

The demo uses a reference gene expression signature matrix:

- rows: genes
- columns: compounds/samples

Students run the notebook step by step. They first inspect the data shape and head, then compute compound-compound correlations, cluster compounds with DBSCAN, extract varimax-rotated latent factors, and finally inspect the factor where `estradiol` has a high score.

Expected interpretation:

- high side of the estradiol-selected factor: estrogen-like compounds such as estradiol, estrone, estriol, equilin, estropipate, dienestrol, and diethylstilbestrol
- low side of the same factor: anti-estrogen-related compounds such as tamoxifen, raloxifene, fulvestrant, and clomifene

The teaching focus is not coding. Students mainly:

- run cells,
- inspect the data matrix,
- look at compound-compound correlation,
- interpret DBSCAN clusters,
- inspect factor scores,
- compare `n_components=10` and `n_components=40`,
- discuss why factor number, rotation, and sign orientation matter.

## Notes on factor extraction

For Colab speed, the default factor extraction is implemented as:

1. standardize genes across compounds,
2. apply randomized PCA,
3. apply orthogonal varimax rotation,
4. select the factor with the largest absolute `estradiol` score,
5. flip the sign so that `estradiol` is positive.

This is used as a fast exploratory factor-analysis-style workflow. It is intended for lecture demonstration rather than final scientific inference.

## Instructor notes

The notebook should be treated as a stable lecture handout. It is intentionally stepwise: students should execute one cell at a time and interpret each output before moving on. Development should be done in `src/latent_demo`, and the notebook should remain thin.

Recommended release workflow:

1. Confirm that tests pass.
2. Push to GitHub under `mizuno-group/latent-demo`.
3. Create a release tag such as `v0.2.1`.
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
