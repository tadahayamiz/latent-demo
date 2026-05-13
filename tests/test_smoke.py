from latent_demo.config import list_scenarios, load_scenario
from latent_demo.data import generate_synthetic_data, prepare_cmap_data
from latent_demo.cmap import match_samples, summarize_terms
from latent_demo.models import (
    cluster_compounds_dbscan,
    compare_cmap_factor_numbers,
    compute_compound_correlation,
    fit_cmap_varimax_factors,
    fit_latent_model,
)


def test_original_synthetic_demo_still_runs():
    cfg = load_scenario("pharm_toxicity_basic")
    generated = generate_synthetic_data(cfg)
    fit = fit_latent_model(
        generated.observed,
        generated.variable_ids,
        model="factor_analysis",
        n_factors=3,
        standardize=True,
    )
    assert fit.scores.shape[1] == 3
    assert fit.reconstruction_mse >= 0


def test_cmap_data_is_packaged_and_loadable():
    cmap = prepare_cmap_data(n_top_genes=120)
    assert cmap.gene_by_compound.shape[0] == 120
    assert "estradiol" in cmap.compound_by_gene.index
    assert cmap.compound_by_gene.shape[0] > 100


def test_cmap_correlation_and_dbscan():
    cmap = prepare_cmap_data(n_top_genes=120)
    corr = compute_compound_correlation(cmap.compound_by_gene)
    assert corr.shape[0] == cmap.compound_by_gene.shape[0]
    clusters = cluster_compounds_dbscan(corr, eps=0.6, min_samples=3)
    assert "cluster" in clusters.columns
    assert clusters.shape[0] == corr.shape[0]


def test_cmap_varimax_factor_ranking():
    cmap = prepare_cmap_data(n_top_genes=200)
    result = fit_cmap_varimax_factors(
        cmap.compound_by_gene,
        n_components=5,
        target_sample="estradiol",
        n_top_genes=200,
    )
    assert result.scores.shape[1] == 5
    assert result.selected_factor in result.scores.columns
    assert "estradiol" in result.ranked_scores.index


def test_cmap_factor_number_comparison():
    cmap = prepare_cmap_data(n_top_genes=120)
    summary = compare_cmap_factor_numbers(cmap.compound_by_gene, component_grid=(2, 5))
    assert list(summary["n_components"]) == [2, 5]
    assert summary["test_log_likelihood"].notna().all()


def test_cmap_stepwise_helpers():
    cmap = prepare_cmap_data(n_top_genes=120)
    corr = compute_compound_correlation(cmap.compound_by_gene)
    clusters = cluster_compounds_dbscan(corr, eps=0.6, min_samples=3)
    result = fit_cmap_varimax_factors(
        cmap.compound_by_gene,
        n_components=5,
        target_sample="estradiol",
        n_top_genes=120,
    )
    hits = match_samples(result.ranked_scores.index, ["estradiol"])
    assert hits == ["estradiol"]
    summary = summarize_terms(result.ranked_scores)
    assert "category" in summary.columns
    assert clusters.shape[0] == corr.shape[0]
