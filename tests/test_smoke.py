from pathlib import Path

import latent_demo
from latent_demo.cmap import (
    ESTROGEN_TERMS,
    cluster_compounds_dbscan,
    compare_cmap_factor_numbers,
    compute_compound_correlation,
    fit_cmap_varimax_factors,
    match_samples,
    plot_dbscan_tsne_scatter,
    rank_target_correlations,
    prepare_cmap_data,
    summarize_terms,
)


def test_public_api_imports():
    assert latent_demo.__version__ == "0.2.2"
    assert callable(prepare_cmap_data)
    assert "estradiol" in ESTROGEN_TERMS


def test_cmap_data_is_packaged_and_loadable():
    cmap = prepare_cmap_data(n_top_genes=20)
    assert cmap.gene_by_compound.shape[0] == 20
    assert "estradiol" in cmap.compound_by_gene.index
    assert cmap.compound_by_gene.shape[0] > 100


def test_cmap_correlation_and_dbscan():
    cmap = prepare_cmap_data(n_top_genes=20)
    corr = compute_compound_correlation(cmap.compound_by_gene)
    ranked = rank_target_correlations(corr, target_sample="estradiol")
    clusters = cluster_compounds_dbscan(corr, eps=0.6, min_samples=3)
    assert corr.shape[0] == cmap.compound_by_gene.shape[0]
    assert clusters.shape[0] == corr.shape[0]
    assert "cluster" in clusters.columns
    assert "correlation" in ranked.columns


def test_cmap_tsne_plot_smoke():
    cmap = prepare_cmap_data(n_top_genes=20)
    corr = compute_compound_correlation(cmap.compound_by_gene)
    clusters = cluster_compounds_dbscan(corr, eps=0.6, min_samples=3)
    fig = plot_dbscan_tsne_scatter(corr, clusters, perplexity=5, highlight_terms=["estradiol"])
    assert fig is not None


def test_cmap_varimax_factor_ranking():
    cmap = prepare_cmap_data(n_top_genes=30)
    result = fit_cmap_varimax_factors(
        cmap.compound_by_gene,
        n_components=2,
        target_sample="estradiol",
        n_top_genes=30,
    )
    assert result.scores.shape[1] == 2
    assert result.selected_factor in result.scores.columns
    assert "estradiol" in result.ranked_scores.index


def test_cmap_factor_number_comparison():
    cmap = prepare_cmap_data(n_top_genes=20)
    summary = compare_cmap_factor_numbers(cmap.compound_by_gene, component_grid=(2,))
    assert list(summary["n_components"]) == [2]
    assert summary["test_log_likelihood"].notna().all()


def test_cmap_term_helpers():
    cmap = prepare_cmap_data(n_top_genes=20)
    result = fit_cmap_varimax_factors(cmap.compound_by_gene, n_components=2, target_sample="estradiol")
    assert match_samples(result.ranked_scores.index, ["estradiol"]) == ["estradiol"]
    summary = summarize_terms(result.ranked_scores)
    assert "category" in summary.columns


def test_old_toxicology_scenarios_removed():
    package_root = Path(latent_demo.__file__).resolve().parent
    assert not (package_root / "scenarios").exists()
