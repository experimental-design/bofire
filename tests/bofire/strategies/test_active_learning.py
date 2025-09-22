import pandas as pd
import pytest

import bofire.strategies.api as strategies
from bofire.benchmarks.multi import DTLZ2
from bofire.data_models.acquisition_functions.api import qNegIntPosVar
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import ActiveLearningStrategy
from bofire.data_models.surrogates.api import BotorchSurrogates, SingleTaskGPSurrogate
from bofire.strategies.predictives.active_learning import (
    ActiveLearningStrategy as ActiveLearningStrategyImpl,
)


def test_active_learning():
    """Tests the initialization of the ActiveLearningStrategy.
    This is done for the most complicated case meaning a multi-objective scenario with
    a unique weight for each output feature.
    """
    benchmark = DTLZ2(dim=3)
    output_keys = benchmark.domain.outputs.get_keys()
    weights = {
        output_keys[0]: 0.3,
        output_keys[1]: 0.7,
    }
    aqcf_data_model = qNegIntPosVar(weights=weights)
    data_model = ActiveLearningStrategy(
        domain=benchmark.domain,
        surrogate_specs=BotorchSurrogates(
            surrogates=[
                SingleTaskGPSurrogate(
                    inputs=benchmark.domain.inputs,
                    outputs=Outputs(features=[benchmark.domain.outputs[0]]),
                ),
                SingleTaskGPSurrogate(
                    inputs=benchmark.domain.inputs,
                    outputs=Outputs(features=[benchmark.domain.outputs[1]]),
                ),
            ],
        ),
        acquisition_function=aqcf_data_model,
    )
    initial_points = benchmark.domain.inputs.sample(10)
    initial_experiments = pd.concat(
        [initial_points, benchmark.f(initial_points)],
        axis=1,
    )
    recommender = strategies.map(data_model=data_model)
    recommender.tell(initial_experiments)  # Check whether the model can be trained.
    # Cast to implementation class to access protected method
    assert isinstance(recommender, ActiveLearningStrategyImpl)
    acqf = recommender._get_acqfs(1)[
        0
    ]  # Check if an instance of the acqf can be created.
    weight_list = []
    [
        weight_list.append(aqcf_data_model.weights.get(key))
        for key in benchmark.domain.outputs.get_keys()
    ]
    assert (
        weight_list == acqf.posterior_transform.weights.tolist()
    )  # Check whether the weights in the posterior_transfrom are set up correctly.
    _ = recommender.ask(2)  # Check whether the optimization of the acqf works.


def test_active_learning_with_categorical():
    """Tests the ActiveLearningStrategy with categorical inputs.
    This test ensures that the fix for handling categorical parameters works correctly.
    """
    # Create a domain with mixed continuous and categorical inputs
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="temperature", bounds=(20, 100)),
                ContinuousInput(key="time", bounds=(0.5, 24)),
                CategoricalInput(key="catalyst", categories=["A", "B", "C"]),
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key="yield", objective=MaximizeObjective(w=1.0)),
            ]
        ),
    )

    # Create the ActiveLearningStrategy
    data_model = ActiveLearningStrategy(
        domain=domain,
        acquisition_function=qNegIntPosVar(
            n_mc_samples=4
        ),  # Use small number for faster test
    )

    # Generate initial experiments
    initial_data = pd.DataFrame(
        {
            "temperature": [25.0, 50.0, 75.0, 90.0],
            "time": [1.0, 4.0, 8.0, 12.0],
            "catalyst": ["A", "B", "C", "A"],
            "yield": [45.0, 68.0, 82.0, 75.0],
            "valid_yield": [1, 1, 1, 1],
        }
    )

    # Map to strategy and tell initial data
    recommender = strategies.map(data_model=data_model)
    recommender.tell(initial_data)

    # Verify the strategy is set up correctly and has set up the acquisition function
    assert isinstance(recommender, ActiveLearningStrategyImpl)
    acqf = recommender._get_acqfs(1)[0]
    assert acqf is not None

    # Ask for new recommendations - this exercises the full pipeline
    recommendations = recommender.ask(candidate_count=1)

    # Verify we get valid recommendations
    assert len(recommendations) == 1
    assert "temperature" in recommendations.columns
    assert "time" in recommendations.columns
    assert "catalyst" in recommendations.columns
    assert recommendations["catalyst"].iloc[0] in ["A", "B", "C"]


def test_active_learning_categorical_multiobjective_weighted():
    """Test Active Learning with categoricals and multiobjective optimization with weights.
    Uses two continuous inputs, two categorical inputs (3 categories each),
    and two outputs with weights (yield: 1.0, impurity: 0.5).
    """
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="temperature", bounds=(20, 100)),
                ContinuousInput(key="pressure", bounds=(1, 10)),
                CategoricalInput(key="catalyst", categories=["A", "B", "C"]),
                CategoricalInput(
                    key="solvent", categories=["water", "ethanol", "methanol"]
                ),
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key="yield", objective=MaximizeObjective(w=1.0)),
                ContinuousOutput(key="impurity", objective=MaximizeObjective(w=0.5)),
            ]
        ),
    )

    # Set up weights for the acquisition function
    weights = {
        "yield": 1.0,
        "impurity": 0.5,
    }

    # Create the ActiveLearningStrategy with weighted multiobjective
    data_model = ActiveLearningStrategy(
        domain=domain,
        surrogate_specs=BotorchSurrogates(
            surrogates=[
                SingleTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=Outputs(features=[domain.outputs.get_by_key("yield")]),
                ),
                SingleTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=Outputs(features=[domain.outputs.get_by_key("impurity")]),
                ),
            ],
        ),
        acquisition_function=qNegIntPosVar(weights=weights, n_mc_samples=4),
    )

    # Generate initial experiments with both outputs
    initial_data = pd.DataFrame(
        {
            "temperature": [25.0, 50.0, 75.0, 90.0, 35.0],
            "pressure": [2.0, 4.0, 6.0, 8.0, 3.0],
            "catalyst": ["A", "B", "C", "A", "B"],
            "solvent": ["water", "ethanol", "methanol", "water", "methanol"],
            "yield": [45.0, 68.0, 82.0, 75.0, 55.0],
            "impurity": [12.0, 8.0, 5.0, 7.0, 10.0],
            "valid_yield": [1, 1, 1, 1, 1],
            "valid_impurity": [1, 1, 1, 1, 1],
        }
    )

    # Map to strategy and tell initial data
    recommender = strategies.map(data_model=data_model)
    recommender.tell(initial_data)

    # Verify the acquisition function is set up correctly
    assert isinstance(recommender, ActiveLearningStrategyImpl)
    acqf = recommender._get_acqfs(1)[0]
    assert acqf is not None

    # Check that weights are properly configured
    weight_list = []
    for key in domain.outputs.get_keys():
        weight_list.append(weights.get(key))
    assert weight_list == acqf.posterior_transform.weights.tolist()

    # Ask for new recommendations
    recommendations = recommender.ask(candidate_count=1)

    # Verify we get valid recommendations
    assert len(recommendations) == 1
    assert "temperature" in recommendations.columns
    assert "pressure" in recommendations.columns
    assert "catalyst" in recommendations.columns
    assert "solvent" in recommendations.columns
    assert recommendations["catalyst"].iloc[0] in ["A", "B", "C"]
    assert recommendations["solvent"].iloc[0] in ["water", "ethanol", "methanol"]


@pytest.mark.slow
def test_active_learning_multiple_categorical():
    """Test Active Learning with multiple categorical inputs."""
    domain = Domain(
        inputs=Inputs(
            features=[
                CategoricalInput(
                    key="solvent", categories=["water", "ethanol", "methanol"]
                ),
                CategoricalInput(key="catalyst", categories=["Pd", "Pt", "Ru", "Ni"]),
                CategoricalInput(key="base", categories=["NaOH", "KOH"]),
                ContinuousInput(key="temperature", bounds=(20, 100)),
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key="conversion", objective=MaximizeObjective(w=1.0)),
            ]
        ),
    )

    data_model = ActiveLearningStrategy(
        domain=domain,
        acquisition_function=qNegIntPosVar(n_mc_samples=4),
    )

    initial_data = pd.DataFrame(
        {
            "solvent": ["water", "ethanol", "methanol", "water", "ethanol"],
            "catalyst": ["Pd", "Pt", "Ru", "Ni", "Pd"],
            "base": ["NaOH", "KOH", "NaOH", "KOH", "NaOH"],
            "temperature": [25.0, 40.0, 60.0, 80.0, 95.0],
            "conversion": [45.0, 58.0, 72.0, 68.0, 81.0],
            "valid_conversion": [1, 1, 1, 1, 1],
        }
    )

    recommender = strategies.map(data_model=data_model)
    recommender.tell(initial_data)

    recommendations = recommender.ask(candidate_count=1)
    assert len(recommendations) == 1
    assert all(
        rec in ["water", "ethanol", "methanol"] for rec in recommendations["solvent"]
    )
    assert all(rec in ["Pd", "Pt", "Ru", "Ni"] for rec in recommendations["catalyst"])
    assert all(rec in ["NaOH", "KOH"] for rec in recommendations["base"])


@pytest.mark.slow
def test_active_learning_only_categorical():
    """Test Active Learning with only categorical inputs (no continuous)."""
    domain = Domain(
        inputs=Inputs(
            features=[
                CategoricalInput(key="material", categories=["A", "B", "C", "D"]),
                CategoricalInput(key="method", categories=["X", "Y", "Z"]),
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key="quality", objective=MaximizeObjective(w=1.0)),
            ]
        ),
    )

    data_model = ActiveLearningStrategy(
        domain=domain,
        acquisition_function=qNegIntPosVar(n_mc_samples=4),
    )

    initial_data = pd.DataFrame(
        {
            "material": ["A", "B", "C", "D", "A", "B"],
            "method": ["X", "Y", "Z", "X", "Y", "Z"],
            "quality": [3.2, 4.1, 5.5, 4.8, 3.9, 4.5],
            "valid_quality": [1, 1, 1, 1, 1, 1],
        }
    )

    recommender = strategies.map(data_model=data_model)
    recommender.tell(initial_data)

    # This should work even with only categorical inputs
    recommendations = recommender.ask(candidate_count=1)
    assert len(recommendations) == 1
    assert recommendations["material"].iloc[0] in ["A", "B", "C", "D"]
    assert recommendations["method"].iloc[0] in ["X", "Y", "Z"]


@pytest.mark.slow
def test_active_learning_check_preds_have_high_uncertainties():
    """Test that Active Learning predictions have high uncertainties.
    This test evaluates that the acquisition function properly focuses on
    regions with high uncertainty (high standard deviation).
    """
    import numpy as np

    # Create a realistic domain with mixed inputs
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="temperature", bounds=(20, 120)),
                ContinuousInput(key="pressure", bounds=(1, 10)),
                ContinuousInput(key="flow_rate", bounds=(0.1, 5.0)),
                CategoricalInput(key="catalyst", categories=["A", "B", "C"]),
                CategoricalInput(
                    key="solvent", categories=["solvent_A", "solvent_B", "solvent_C"]
                ),
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key="yield", objective=MaximizeObjective(w=1.0)),
            ]
        ),
    )

    # Create 10 initial data points with clean linear and quadratic relationships
    # yield = 20 + 0.5*temperature - 0.001*temperature^2 + 2*pressure + catalyst_effect + solvent_effect
    # catalyst effects: A=0, B=5, C=10
    # solvent effects: solvent_A=0, solvent_B=3, solvent_C=6
    np.random.seed(42)  # For reproducibility

    temperatures = [25, 40, 60, 80, 100, 30, 55, 75, 95, 110]
    pressures = [1.5, 3.0, 5.0, 7.0, 9.0, 2.0, 4.5, 6.5, 8.5, 9.5]
    flow_rates = [0.2, 0.8, 1.5, 2.2, 3.0, 0.5, 1.2, 2.0, 2.8, 4.0]
    catalysts = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"]
    solvents = [
        "solvent_A",
        "solvent_B",
        "solvent_C",
        "solvent_A",
        "solvent_B",
        "solvent_C",
        "solvent_A",
        "solvent_B",
        "solvent_C",
        "solvent_A",
    ]

    # Calculate yields with deterministic formula
    yields = []
    catalyst_effects = {"A": 0, "B": 5, "C": 10}
    solvent_effects = {"solvent_A": 0, "solvent_B": 3, "solvent_C": 6}

    for i in range(10):
        base_yield = (
            20
            + 0.5 * temperatures[i]
            - 0.001 * temperatures[i] ** 2
            + 2 * pressures[i]
            + catalyst_effects[catalysts[i]]
            + solvent_effects[solvents[i]]
        )
        # Add small random noise
        yields.append(base_yield + np.random.normal(0, 0.5))

    initial_data = pd.DataFrame(
        {
            "temperature": temperatures,
            "pressure": pressures,
            "flow_rate": flow_rates,
            "catalyst": catalysts,
            "solvent": solvents,
            "yield": yields,
            "valid_yield": [1] * 10,
        }
    )

    # Create strategy with adequate MC samples
    data_model = ActiveLearningStrategy(
        domain=domain,
        acquisition_function=qNegIntPosVar(n_mc_samples=64),
    )

    recommender = strategies.map(data_model=data_model)

    # Tell the strategy about existing data
    recommender.tell(initial_data)

    # Ask for 2 candidates
    recommendations = recommender.ask(candidate_count=2)

    assert len(recommendations) == 2

    # Verify categorical values are valid
    for idx in range(2):
        assert recommendations["catalyst"].iloc[idx] in ["A", "B", "C"]
        assert recommendations["solvent"].iloc[idx] in [
            "solvent_A",
            "solvent_B",
            "solvent_C",
        ]

    # Now evaluate many points to check that recommendations have high uncertainty
    # Generate test points using domain.inputs.sample
    test_samples = domain.inputs.sample(100)

    # Predict on test samples to get uncertainties
    predictions = recommender.predict(test_samples)

    # Get standard deviations for all test samples
    std_devs = predictions["yield_sd"].values

    # Get standard deviations for our recommendations
    rec_predictions = recommender.predict(
        recommendations[["temperature", "pressure", "flow_rate", "catalyst", "solvent"]]
    )
    rec_stds = rec_predictions["yield_sd"].values

    # Active Learning should select points with high uncertainty
    # Check that the recommended points are in the upper percentile of standard deviations
    percentile_50 = np.percentile(std_devs, 50)

    # At least one recommendation should have high uncertainty (above median)
    assert any(std >= percentile_50 for std in rec_stds), (
        f"Active Learning should focus on uncertain regions. "
        f"Recommendation stds: {rec_stds}, median: {percentile_50}"
    )

    # Additional check: recommendations should explore new regions
    for idx in range(len(recommendations)):
        rec_point = recommendations.iloc[idx]

        min_distance = float("inf")
        for _, existing in initial_data.iterrows():
            if (
                rec_point["catalyst"] == existing["catalyst"]
                and rec_point["solvent"] == existing["solvent"]
            ):
                # Calculate distance in continuous space
                temp_diff = abs(rec_point["temperature"] - existing["temperature"])
                pressure_diff = abs(rec_point["pressure"] - existing["pressure"])
                flow_diff = abs(rec_point["flow_rate"] - existing["flow_rate"])

                # Normalized distance
                distance = (
                    (temp_diff / 100) ** 2
                    + (pressure_diff / 9) ** 2
                    + (flow_diff / 4.9) ** 2
                )
                min_distance = min(min_distance, distance)

        # Recommendations should have some distance from existing points
        assert (
            min_distance > 0.001
        ), "Recommendations should explore new regions, not repeat existing experiments"
