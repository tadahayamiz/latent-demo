# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

High-level workflows used from the Colab notebook.

@author: tadahaya
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from matplotlib import pyplot as plt

from .config import load_scenario
from .data import GeneratedData, generate_synthetic_data
from .layout import display_dataframe, display_markdown, show_question
from .metrics import evaluate_factor_numbers, latent_alignment
from .models import LatentModelResult, fit_latent_model
from .plots import (
    plot_alignment,
    plot_correlation_heatmap,
    plot_loadings_heatmap,
    plot_model_selection,
    plot_scores,
    plot_unique_noise,
)


@dataclass
class DemoResult:
    cfg: dict[str, Any]
    generated: GeneratedData
    fit: LatentModelResult
    alignment: Any | None = None


def run_latent_toxicity_demo(
    scenario: str = "pharm_toxicity_basic",
    model: str = "factor_analysis",
    n_factors: int = 3,
    standardize: bool | None = None,
    show_truth: bool = False,
    display: bool = True,
) -> DemoResult:
    """Run the main latent toxicity demo.

    This function is intentionally high-level so that students can focus on
    interpreting figures rather than writing analysis code.
    """
    cfg = load_scenario(scenario)
    generated = generate_synthetic_data(cfg)
    if standardize is None:
        standardize = bool(cfg.get("dataset", {}).get("standardize", True))

    fit = fit_latent_model(
        generated.observed,
        generated.variable_ids,
        model=model,
        n_factors=n_factors,
        standardize=standardize,
        random_state=int(cfg.get("dataset", {}).get("seed", 42)),
    )

    alignment = None
    if show_truth:
        alignment = latent_alignment(fit.scores, generated.latent, generated.factor_ids)

    if display:
        title = cfg.get("scenario", {}).get("title_ja", "Latent variable demo")
        display_markdown(f"## {title}\n\n{cfg.get('scenario', {}).get('description_ja', '')}")

        display_markdown(
            f"### 1. 観測できるデータ\n"
            f"以下は学生が最初に見る検査値データです。真の潜在状態はまだ見せない想定です。"
        )
        display_dataframe(generated.observed, max_rows=8)
        show_question("この表だけから、各サンプルの『肝障害状態』『炎症状態』『骨髄抑制状態』はわかるでしょうか？")

        display_markdown("### 2. 相関を見る")
        plot_correlation_heatmap(generated.observed, generated.variable_ids, title="Observed variables: correlation")
        plt.show()
        show_question("一緒に動いている検査値の組み合わせはどれですか？")

        display_markdown(f"### 3. {model} による潜在空間")
        plot_scores(fit.scores, generated.labels, show_truth=show_truth, title=f"Estimated latent scores ({model}, q={n_factors})")
        plt.show()
        display_markdown(
            f"- standardize = `{standardize}`\n"
            f"- reconstruction MSE = `{fit.reconstruction_mse:.4f}`\n"
            f"- average log-likelihood = `{fit.log_likelihood if fit.log_likelihood is not None else 'N/A'}`"
        )

        display_markdown("### 4. 負荷量から潜在因子を解釈する")
        display_dataframe(fit.loadings.round(3), max_rows=len(fit.loadings))
        plot_loadings_heatmap(fit.loadings, title="Estimated loadings")
        plt.show()
        show_question("LV1, LV2, LV3 に薬学的な名前をつけるとしたら何でしょうか？")

        if fit.unique_noise is not None:
            display_markdown("### 5. 変数ごとの固有ノイズ")
            display_dataframe(fit.unique_noise.sort_values(ascending=False).round(3).to_frame(), max_rows=len(fit.unique_noise))
            plot_unique_noise(fit.unique_noise.sort_values(ascending=False), title="Estimated unique noise")
            plt.show()
            show_question("どの検査値は共通因子よりも固有ノイズが大きそうでしょうか？")

        if show_truth and alignment is not None:
            display_markdown("### 6. 真の潜在因子を開示する")
            display_markdown("人工データなので、最後にだけ真の潜在因子との対応を確認できます。")
            display_dataframe(alignment.round(3), max_rows=len(alignment))
            plot_alignment(alignment, title="Estimated axes vs. true hidden factors")
            plt.show()

    return DemoResult(cfg=cfg, generated=generated, fit=fit, alignment=alignment)


def compare_factor_numbers(
    scenario: str = "pharm_toxicity_basic",
    model: str = "factor_analysis",
    max_factors: int = 5,
    standardize: bool = True,
    display: bool = True,
):
    """Compare reconstruction error and held-out likelihood for q=1..max_factors."""
    cfg = load_scenario(scenario)
    generated = generate_synthetic_data(cfg)
    summary = evaluate_factor_numbers(
        generated.observed,
        generated.variable_ids,
        model=model,
        max_factors=max_factors,
        standardize=standardize,
        random_state=int(cfg.get("dataset", {}).get("seed", 42)),
    )
    if display:
        display_markdown("## 発展：因子数を変えると何が起こるか")
        display_dataframe(summary.round(4), max_rows=len(summary))
        plot_model_selection(summary)
        plt.show()
        show_question("訓練誤差だけでなく、テスト尤度や解釈性を考えると、因子数はいくつがよさそうですか？")
    return summary


# -----------------------------------------------------------------------------
# CMap / estradiol signature demo workflows
# -----------------------------------------------------------------------------

from .data import prepare_cmap_data
from .models import (
    cluster_compounds_dbscan,
    compare_cmap_factor_numbers as _compare_cmap_factor_numbers,
    compute_compound_correlation,
    fit_cmap_varimax_factors,
)
from .plots import (
    plot_cmap_factor_number_summary,
    plot_compound_correlation_heatmap,
    plot_dbscan_pca_scatter,
    plot_ranked_factor_scores,
)


ESTROGEN_TERMS = [
    "estradiol",
    "estrone",
    "estriol",
    "estropipate",
    "diethylstilbestrol",
    "dienestrol",
    "equilin",
    "prasterone",
]

ANTI_ESTROGEN_TERMS = [
    "tamoxifen",
    "raloxifene",
    "fulvestrant",
    "clomifene",
    "mifepristone",
]


def _subset_by_terms(index, terms: list[str]) -> list[str]:
    return [x for x in index if any(term.lower() in x.lower() for term in terms)]


def run_cmap_estradiol_demo(
    n_top_genes: int = 3000,
    n_components: int = 40,
    target_sample: str = "estradiol",
    top_n: int = 20,
    dbscan_eps: float = 0.6,
    dbscan_min_samples: int = 3,
    show_plots: bool = True,
):
    """Run the CMap estradiol-factor demo.

    The default settings are chosen so that an estradiol-high factor tends to
    rank estrogen-like compounds high and anti-estrogen compounds low.
    """
    display_markdown("## 1. セットアップ：パッケージ内蔵データを読み込む")
    cmap = prepare_cmap_data(n_top_genes=n_top_genes)
    display_markdown(
        f"- 使用データ: packaged `ref_cmap.csv`  \\n"
        f"- genes used: `{cmap.gene_by_compound.shape[0]}`  \\n"
        f"- compounds/samples: `{cmap.gene_by_compound.shape[1]}`"
    )

    display_markdown("## 2. データサイズと先頭を確認する")
    display_dataframe(cmap.gene_by_compound.iloc[:5, :8], max_rows=5)
    show_question("行が遺伝子、列が化合物です。各値は化合物処理に伴う発現変化シグネチャとみなせます。")

    display_markdown("## 3. 化合物間の相関を見る")
    correlation = compute_compound_correlation(cmap.compound_by_gene)
    display_dataframe(correlation.iloc[:8, :8].round(3), max_rows=8)
    if show_plots:
        plot_compound_correlation_heatmap(correlation, title="Compound-compound correlation from expression signatures")
        plt.show()
    show_question("相関が高い化合物は、遺伝子発現シグネチャが似ている化合物です。ぼんやりした構造は見えるでしょうか？")

    display_markdown("## 4. DBSCANで近い化合物の塊を見る")
    clusters = cluster_compounds_dbscan(correlation, eps=dbscan_eps, min_samples=dbscan_min_samples)
    cluster_summary = clusters["cluster"].value_counts().sort_index().rename_axis("cluster").to_frame("n_samples")
    display_dataframe(cluster_summary, max_rows=len(cluster_summary))
    if show_plots:
        plot_dbscan_pca_scatter(
            cmap.compound_by_gene,
            clusters,
            highlight_terms=["estradiol", "estrone", "estriol", "tamoxifen", "raloxifene", "fulvestrant", "clomifene"],
            title=f"DBSCAN on correlation distance (eps={dbscan_eps}, min_samples={dbscan_min_samples})",
        )
        plt.show()

    estrogen_hits = _subset_by_terms(cmap.compound_by_gene.index, ESTROGEN_TERMS)
    anti_hits = _subset_by_terms(cmap.compound_by_gene.index, ANTI_ESTROGEN_TERMS)
    display_markdown("### estrogen / anti-estrogen 関連化合物のDBSCAN cluster")
    if estrogen_hits or anti_hits:
        key_clusters = clusters.loc[estrogen_hits + anti_hits].copy()
        key_clusters["category"] = ["estrogen-like"] * len(estrogen_hits) + ["anti-estrogen"] * len(anti_hits)
        display_dataframe(key_clusters, max_rows=len(key_clusters))

    display_markdown("## 5. Varimax因子抽出で estradiol-high factor を見る")
    display_markdown(
        "ここでは Colab 上での速度を優先し、PCAによる初期抽出に varimax 回転をかける高速な探索的因子抽出を用います。"
    )
    factor_result = fit_cmap_varimax_factors(
        cmap.compound_by_gene,
        n_components=n_components,
        target_sample=target_sample,
        n_top_genes=n_top_genes,
        random_state=0,
        select_mode="max_abs",
    )
    display_markdown(
        f"- selected factor: `{factor_result.selected_factor}`  \\n"
        f"- {target_sample} score: `{factor_result.target_score:.3f}`  \\n"
        f"- n_components: `{n_components}`"
    )

    top_samples = factor_result.ranked_scores.head(top_n)
    bottom_samples = factor_result.ranked_scores.tail(top_n).sort_values("factor_score", ascending=True)
    display_markdown(f"### Top {top_n}: estradiol-high factor")
    display_dataframe(top_samples.round(3), max_rows=top_n)
    display_markdown(f"### Bottom {top_n}: opposite side of estradiol-high factor")
    display_dataframe(bottom_samples.round(3), max_rows=top_n)
    if show_plots:
        plot_ranked_factor_scores(
            factor_result.ranked_scores,
            top_n=top_n,
            title=f"Samples ranked by {factor_result.selected_factor} score (target={target_sample})",
        )
        plt.show()

    display_markdown("### estrogen-like / anti-estrogen 関連化合物のスコア")
    if estrogen_hits or anti_hits:
        key_scores = factor_result.ranked_scores.loc[estrogen_hits + anti_hits].copy()
        key_scores["category"] = ["estrogen-like"] * len(estrogen_hits) + ["anti-estrogen"] * len(anti_hits)
        key_scores = key_scores.sort_values("factor_score", ascending=False)
        display_dataframe(key_scores.round(3), max_rows=len(key_scores))
    show_question("Top側に estrogen-like な化合物、Bottom側に anti-estrogen が来るでしょうか？")

    return {
        "cmap": cmap,
        "correlation": correlation,
        "clusters": clusters,
        "factor_result": factor_result,
        "top_samples": top_samples,
        "bottom_samples": bottom_samples,
    }


def compare_cmap_factor_numbers_workflow(
    n_top_genes: int = 3000,
    component_grid: list[int] | tuple[int, ...] = (5, 10, 20, 40, 60),
    show_plots: bool = True,
):
    """Compare latent factor numbers for the CMap demo."""
    display_markdown("## 発展：因子数の妥当性を尤度で眺める")
    cmap = prepare_cmap_data(n_top_genes=n_top_genes)
    summary = _compare_cmap_factor_numbers(cmap.compound_by_gene, component_grid=component_grid)
    display_dataframe(summary.round(4), max_rows=len(summary))
    if show_plots:
        plot_cmap_factor_number_summary(summary, title="PPCA-like held-out likelihood by component number")
        plt.show()
    show_question("再構成誤差だけでなく、テスト対数尤度や解釈性を考えると、因子数はいくつがよさそうでしょうか？")
    return summary
