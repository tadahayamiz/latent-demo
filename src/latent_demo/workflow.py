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
