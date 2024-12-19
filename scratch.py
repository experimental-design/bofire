import pandas as pd

import bofire.strategies.api as strategies
import bofire.surrogates.api as surrogates
from bofire.data_models.domain import api as domain_api
from bofire.data_models.features import api as features_api
from bofire.data_models.kernels import api as kernels_api
from bofire.data_models.molfeatures import api as molfeatures_api
from bofire.data_models.priors.api import HVARFNER_LENGTHSCALE_PRIOR
from bofire.data_models.strategies import api as strategies_api
from bofire.data_models.surrogates import api as surrogates_api


def test_SingleTaskGPModel_mixed_features():
    """test that we can use a single task gp with mixed features"""
    inputs = domain_api.Inputs(
        features=[
            features_api.ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            features_api.CategoricalInput(key="x_cat_1", categories=["mama", "papa"]),
            features_api.CategoricalInput(key="x_cat_2", categories=["cat", "dog"]),
        ]
    )
    outputs = domain_api.Outputs(features=[features_api.ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat_1 == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat_1 == "papa", "y"] /= 2.0
    experiments.loc[experiments.x_cat_2 == "cat", "y"] *= -2.0
    experiments.loc[experiments.x_cat_2 == "dog", "y"] /= -5.0
    experiments["valid_y"] = 1

    gp_data = surrogates_api.SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernels_api.AdditiveKernel(
            kernels=[
                kernels_api.HammingDistanceKernel(
                    ard=True,
                    features=["x_cat_1", "x_cat_2"],
                ),
                kernels_api.RBFKernel(
                    ard=True,
                    lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                    features=[f"x_{i+1}" for i in range(2)],
                ),
            ]
        ),
    )

    gp_mapped = surrogates.map(gp_data)
    assert hasattr(gp_mapped, "fit")
    assert len(gp_mapped.kernel.kernels) == 2
    assert gp_mapped.kernel.kernels[0].features == ["x_cat_1", "x_cat_2"]
    assert gp_mapped.kernel.kernels[1].features == ["x_1", "x_2"]
    gp_mapped.fit(experiments)
    pred = gp_mapped.predict(experiments)
    assert pred.shape == (10, 2)
    assert gp_mapped.model.covar_module.kernels[0].active_dims.tolist() == [2, 3, 4, 5]
    assert gp_mapped.model.covar_module.kernels[1].active_dims.tolist() == [0, 1]


if __name__ == "__main__":
    test_SingleTaskGPModel_mixed_features()


import sys


sys.exit(0)


domain = domain_api.Domain(
    inputs=domain_api.Inputs(
        features=[
            features_api.ContinuousInput(key="x1", bounds=(-1, 1)),
            features_api.ContinuousInput(key="x2", bounds=(-1, 1)),
            features_api.CategoricalMolecularInput(
                key="mol", categories=["CO", "CCO", "CCCO"]
            ),
        ]
    ),
    outputs=domain_api.Outputs(features=[features_api.ContinuousOutput(key="f")]),
)


strategy = strategies.map(
    strategies_api.SoboStrategy(
        domain=domain,
        surrogate_specs=surrogates_api.BotorchSurrogates(
            surrogates=[
                surrogates_api.SingleTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=domain.outputs,
                    input_preprocessing_specs={
                        "mol": molfeatures_api.Fingerprints(),
                    },
                    kernel=kernels_api.AdditiveKernel(
                        kernels=[
                            kernels_api.RBFKernel(
                                ard=True,
                                lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                                features=["x1", "x2"],
                            ),
                            kernels_api.TanimotoKernel(
                                features=["mol"],
                            ),
                        ]
                    ),
                )
            ]
        ),
    )
)


strategy.tell(
    experiments=pd.DataFrame(
        [
            {"x1": 0.2, "x2": 0.4, "mol": "CO", "f": 1.0},
            {"x1": 0.4, "x2": 0.2, "mol": "CCO", "f": 2.0},
            {"x1": 0.6, "x2": 0.6, "mol": "CCCO", "f": 3.0},
        ]
    )
)
candidates = strategy.ask(candidate_count=1)
print(candidates)
