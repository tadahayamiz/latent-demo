# latent-demo

Small Colab demos for latent variables and compound expression signatures.

This repository currently contains one teaching demo:

> **CMap発現シグネチャから target-associated latent factor を探す**

The repository is intentionally small. The notebook is the student-facing entry point, while reusable code lives under `src/latent_demo`.

## Student entry point

Students do **not** need a GitHub account.

Open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizuno-group/latent-demo/blob/v0.2.2/notebooks/01_cmap_estradiol_factor_demo.ipynb)

Recommended instruction for students:

1. Open the Colab link above.
2. Click **Copy to Drive**.
3. Run the cells from top to bottom.
4. Change only the form inputs unless instructed otherwise.

## Development setup in Colab

During development, the setup cell in the notebook can switch GitHub owner, repo, and branch/tag/commit.

The important rule is:

```text
Notebook ref = package install ref
```

For example, if the notebook is opened from `main`, install from `main`. If the notebook is opened from `v0.2.2`, install from `v0.2.2`.

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
│       ├── cmap.py
│       ├── data.py
│       ├── models.py
│       ├── plots.py
│       └── datasets/
│           └── ref_cmap.csv
└── tests/
    └── test_smoke.py
```

## Demo concept

The demo uses a reference gene expression signature matrix:

- rows: genes
- columns: compounds/samples

Students run the notebook step by step. They first inspect the data shape and head, then list compounds whose expression signatures are correlated with a target compound, cluster compounds with DBSCAN, visualize the clusters by t-SNE, extract varimax-rotated latent factors, and inspect the factor where the selected target compound has a high score.

Expected interpretation when `target_sample="estradiol"`:

- high side of the selected factor: estrogen-like compounds such as estradiol, estrone, estriol, equilin, estropipate, dienestrol, and diethylstilbestrol
- low side of the same factor: anti-estrogen-related compounds such as tamoxifen, raloxifene, fulvestrant, and clomifene

The target can also be changed to another compound name such as `tamoxifen` or `dexamethasone`.

The teaching focus is not coding. Students mainly:

- run cells,
- inspect the data matrix,
- inspect target-correlated compounds,
- interpret DBSCAN clusters on a t-SNE map,
- inspect factor scores,
- compare `n_components=10` and `n_components=40`,
- discuss why factor number, rotation, and sign orientation matter.

## Notes on factor extraction

For Colab speed, the default factor extraction is implemented as:

1. standardize genes across compounds,
2. apply randomized PCA,
3. apply orthogonal varimax rotation,
4. select the factor with the largest absolute target-compound score,
5. flip the sign so that the target compound is positive.

This is used as a fast exploratory factor-analysis-style workflow. It is intended for lecture demonstration rather than final scientific inference.

## Instructor notes

The notebook should be treated as a stable lecture handout. It is intentionally stepwise: students should execute one cell at a time and interpret each output before moving on. Development should be done in `src/latent_demo`, and the notebook should remain thin.

Recommended release workflow:

1. Confirm that tests pass.
2. Push to GitHub under `mizuno-group/latent-demo`.
3. During development, test with `GIT_REF="main"` or a development branch in the setup cell.
4. For the final lecture handout, create a stable release tag such as `v0.2.2` and use the tagged Colab URL in slides or LMS.

## Test

```bash
pytest -q
```

## Authors
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- tadahaya[at]gmail.com  
