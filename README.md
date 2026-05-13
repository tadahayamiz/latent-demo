# latent-demo

`latent-demo` は, 潜在変数モデリングを題材にした小規模なColab実習用repoである。

現在は以下の講義実習を含む。

> **CMap発現プロファイルから target-associated latent factor を探す**

## 学生向け入口

以下のリンクから notebook を直接 Colab で開く。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mizuno-group/latent-demo/blob/v0.2.2/notebooks/01_cmap_estradiol_factor_demo.ipynb)

### 手順

1. 上の Colab リンクを開く。
2. **ドライブにコピー** を押す。
3. コピーされた notebook を上から順に実行する。
4. 指示がある場合を除き, フォーム入力だけを変更する。


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

## Test

```bash
pytest -q
```

## References
- [the connectivity map project](https://pubmed.ncbi.nlm.nih.gov/17008526/)
    - Lamb J, Crawford ED, Peck D, Modell JW, Blat IC, Wrobel MJ, Lerner J, Brunet JP, Subramanian A, Ross KN, Reich M, Hieronymus H, Wei G, Armstrong SA, Haggarty SJ, Clemons PA, Wei R, Carr SA, Lander ES, Golub TR. The Connectivity Map: using gene-expression signatures to connect small molecules, genes, and disease. Science. 2006 Sep 29;313(5795):1929-35. doi: 10.1126/science.1132939. PMID: 17008526.

## Authors

- [Tadahaya Mizuno](https://github.com/tadahayamiz)

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- tadahaya[at]gmail.com  
