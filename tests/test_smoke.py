from latent_demo.config import list_scenarios, load_scenario
from latent_demo.data import generate_synthetic_data
from latent_demo.metrics import evaluate_factor_numbers
from latent_demo.models import fit_latent_model
from latent_demo.workflow import compare_factor_numbers, run_latent_toxicity_demo


def test_scenarios_are_available():
    names = list_scenarios()
    assert "pharm_toxicity_basic" in names
    assert "pharm_toxicity_noisy" in names
    assert "pharm_toxicity_missing" in names


def test_generate_and_fit_factor_analysis():
    cfg = load_scenario("pharm_toxicity_basic")
    generated = generate_synthetic_data(cfg)
    assert generated.observed.shape[0] == cfg["dataset"]["n_samples"]
    assert len(generated.variable_ids) >= 6

    fit = fit_latent_model(
        generated.observed,
        generated.variable_ids,
        model="factor_analysis",
        n_factors=3,
        standardize=True,
    )
    assert fit.scores.shape == (cfg["dataset"]["n_samples"], 3)
    assert fit.loadings.shape == (len(generated.variable_ids), 3)
    assert fit.reconstruction_mse >= 0


def test_generate_and_fit_pca():
    cfg = load_scenario("pharm_toxicity_basic")
    generated = generate_synthetic_data(cfg)
    fit = fit_latent_model(
        generated.observed,
        generated.variable_ids,
        model="pca",
        n_factors=2,
        standardize=True,
    )
    assert fit.scores.shape[1] == 2
    assert fit.log_likelihood is not None


def test_factor_number_summary():
    cfg = load_scenario("pharm_toxicity_basic")
    generated = generate_synthetic_data(cfg)
    summary = evaluate_factor_numbers(
        generated.observed,
        generated.variable_ids,
        model="factor_analysis",
        max_factors=3,
        standardize=True,
    )
    assert list(summary["n_factors"]) == [1, 2, 3]
    assert summary["train_reconstruction_mse"].notna().all()


def test_workflows_without_display():
    result = run_latent_toxicity_demo(
        scenario="pharm_toxicity_basic",
        model="factor_analysis",
        n_factors=3,
        standardize=True,
        show_truth=True,
        display=False,
    )
    assert result.alignment is not None

    summary = compare_factor_numbers(
        scenario="pharm_toxicity_basic",
        model="factor_analysis",
        max_factors=3,
        standardize=True,
        display=False,
    )
    assert len(summary) == 3
