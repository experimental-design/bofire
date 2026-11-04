import inspect

import pytest

from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalOutput,
    CategoricalTaskInput,
    ContinuousInput,
    ContinuousOutput,
    ProductFeature,
)
from bofire.data_models.kernels.api import LinearKernel
from bofire.data_models.likelihoods.api import (
    GaussianLikelihood,
    SparseOutlierGaussianLikelihood,
)
from bofire.data_models.means.api import ConstantMean
from bofire.data_models.objectives.api import ConstrainedCategoricalObjective
from bofire.data_models.surrogates import api as specs
from bofire.data_models.surrogates.optimizer import LBFGSB
from bofire.data_models.surrogates.scaler import Normalize
from bofire.surrogates.deterministic import (
    CategoricalDeterministicSurrogate,
    LinearDeterministicSurrogate,
)
from bofire.surrogates.empirical import EmpiricalSurrogate
from bofire.surrogates.fully_bayesian import FullyBayesianSingleTaskGPSurrogate
from bofire.surrogates.map_saas import EnsembleMapSaasSingleTaskGPSurrogate
from bofire.surrogates.mapper import SURROGATE_MAP
from bofire.surrogates.mlp import ClassificationMLPEnsemble, RegressionMLPEnsemble
from bofire.surrogates.multi_task_gp import MultiTaskGPSurrogate
from bofire.surrogates.pairwise_gp import PairwiseGPSurrogate
from bofire.surrogates.random_forest import RandomForestSurrogate
from bofire.surrogates.selection import SelectionSurrogate
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate


FACTORIES = [
    (specs.RegressionMLPEnsemble, RegressionMLPEnsemble, {}),
    (specs.ClassificationMLPEnsemble, ClassificationMLPEnsemble, {}),
    (
        specs.EnsembleMapSaasSingleTaskGPSurrogate,
        EnsembleMapSaasSingleTaskGPSurrogate,
        {},
    ),
    (specs.PairwiseGPSurrogate, PairwiseGPSurrogate, {}),
    (specs.SelectionSurrogate, SelectionSurrogate, {}),
    (specs.SingleTaskGPSurrogate, SingleTaskGPSurrogate, {}),
    (specs.MultiTaskGPSurrogate, MultiTaskGPSurrogate, {}),
    (specs.FullyBayesianSingleTaskGPSurrogate, FullyBayesianSingleTaskGPSurrogate, {}),
    (specs.RandomForestSurrogate, RandomForestSurrogate, {}),
    (specs.EmpiricalSurrogate, EmpiricalSurrogate, {}),
    (
        specs.LinearDeterministicSurrogate,
        LinearDeterministicSurrogate,
        {"coefficients": {"x": 2.0}, "intercept": 1.0},
    ),
    (
        specs.CategoricalDeterministicSurrogate,
        CategoricalDeterministicSurrogate,
        {"mapping": {"a": 1.0, "b": 2.0}},
    ),
]


def features(data_model_cls):
    inputs = [ContinuousInput(key="x", bounds=(0, 1))]
    if data_model_cls is specs.CategoricalDeterministicSurrogate:
        inputs = [CategoricalInput(key="x", categories=["a", "b"])]
    elif data_model_cls is specs.MultiTaskGPSurrogate:
        inputs.append(CategoricalTaskInput(key="task", categories=["a", "b"]))
    if data_model_cls is specs.ClassificationMLPEnsemble:
        return Inputs(features=inputs), Outputs(
            features=[
                CategoricalOutput(
                    key="y",
                    categories=["a", "b"],
                    objective=ConstrainedCategoricalObjective(
                        categories=["a", "b"], desirability=[False, True]
                    ),
                )
            ]
        )
    return Inputs(features=inputs), Outputs(features=[ContinuousOutput(key="y")])


def test_make_covers_registered_surrogates():
    assert {(spec, runtime) for spec, runtime, _ in FACTORIES} == set(
        SURROGATE_MAP.items()
    )
    assert all(
        callable(getattr(runtime, "make", None)) for runtime in SURROGATE_MAP.values()
    )


@pytest.mark.parametrize(
    "data_model_cls,runtime_cls",
    [(data_model_cls, runtime_cls) for data_model_cls, runtime_cls, _ in FACTORIES],
    ids=[runtime_cls.__name__ for _, runtime_cls, _ in FACTORIES],
)
def test_make_defaults_and_signature(data_model_cls, runtime_cls):
    # Catch drift between the duplicated factory and data-model defaults.
    parameters = inspect.signature(runtime_cls.make).parameters
    fields = {
        name: field
        for name, field in data_model_cls.model_fields.items()
        if name != "type"
    }
    assert parameters.keys() == fields.keys(), (
        f"{runtime_cls.__name__}.make parameters differ from data-model fields: "
        f"missing={fields.keys() - parameters.keys()}, "
        f"extra={parameters.keys() - fields.keys()}"
    )
    for name, parameter in parameters.items():
        field = fields[name]
        location = f"{runtime_cls.__name__}.make({name}=...)"
        if field.is_required():
            assert (
                parameter.default is inspect.Parameter.empty
            ), f"{location}: required field has a factory default"
        else:
            assert (
                parameter.default is not inspect.Parameter.empty
            ), f"{location}: missing factory default"
            expected = field.get_default(call_default_factory=True)
            assert parameter.default == expected, (
                f"{location}: default differs from data model; "
                f"make={parameter.default!r}, data_model={expected!r}"
            )


@pytest.mark.parametrize("data_model_cls,runtime_cls,extra", FACTORIES)
def test_make_constructs_model_without_mutating_defaults(
    data_model_cls, runtime_cls, extra, monkeypatch
):
    parameters = inspect.signature(runtime_cls.make).parameters
    inputs, outputs = features(data_model_cls)
    if data_model_cls is specs.SelectionSurrogate:
        extra = {
            "options": [
                specs.SingleTaskGPSurrogate(inputs=inputs, outputs=outputs),
                specs.RandomForestSurrogate(inputs=inputs, outputs=outputs),
            ]
        }
    expected = data_model_cls(inputs=inputs, outputs=outputs, **extra)
    captured = []
    original = runtime_cls.__init__

    def capture(self, data_model, **kwargs):
        captured.append(data_model)
        original(self, data_model=data_model, **kwargs)

    monkeypatch.setattr(runtime_cls, "__init__", capture)
    made = runtime_cls.make(inputs=inputs, outputs=outputs, **extra)
    assert type(made) is runtime_cls
    assert captured[0].model_dump() == expected.model_dump()

    # A serialized configuration can also be supplied as keyword arguments.
    runtime_cls.make(**expected.model_dump(exclude={"type"}))
    assert captured[1].model_dump() == expected.model_dump()

    # Validation must not mutate the concrete objects stored in the signature.
    for name, parameter in parameters.items():
        field = data_model_cls.model_fields[name]
        if not field.is_required():
            assert parameter.default == field.get_default(call_default_factory=True)


@pytest.mark.parametrize(
    "runtime_cls",
    [SingleTaskGPSurrogate, FullyBayesianSingleTaskGPSurrogate, RandomForestSurrogate],
)
def test_explicit_none_disables_scaling(runtime_cls):
    inputs, outputs = features(specs.SingleTaskGPSurrogate)
    assert isinstance(runtime_cls.make(inputs, outputs).scaler, Normalize)
    assert runtime_cls.make(inputs, outputs, scaler=None).scaler is None


def test_multi_task_mean_and_likelihood_forwarding():
    inputs, outputs = features(specs.MultiTaskGPSurrogate)
    mean = ConstantMean()
    likelihood = GaussianLikelihood(noise_constraint=None)
    model = MultiTaskGPSurrogate.make(inputs, outputs, mean=mean, likelihood=likelihood)
    assert model.mean == mean
    assert model.likelihood == likelihood
    assert model.likelihood.noise_constraint is None

    first = MultiTaskGPSurrogate.make(inputs, outputs)
    first.likelihood.likelihood.noise_constraint.lower_bound = 1e-3
    second = MultiTaskGPSurrogate.make(inputs, outputs)
    assert second.likelihood.likelihood.noise_constraint.lower_bound == 1e-4


def test_single_task_current_settings_and_independent_defaults():
    inputs, outputs = features(specs.SingleTaskGPSurrogate)
    first = SingleTaskGPSurrogate.make(inputs, outputs)
    second = SingleTaskGPSurrogate.make(inputs, outputs)
    first.optimizer.maxiter = 7
    first.kernel.ard = False
    first.scaler.features.append("x")
    assert second.optimizer.maxiter is None
    assert second.kernel.ard is True
    assert second.scaler.features == []
    third = SingleTaskGPSurrogate.make(inputs, outputs)
    assert third.optimizer.maxiter is None
    assert third.kernel.ard is True
    assert third.scaler.features == []

    made = SingleTaskGPSurrogate.make(
        inputs,
        outputs,
        kernel=LinearKernel(),
        mean=ConstantMean(),
        likelihood=SparseOutlierGaussianLikelihood(),
        optimizer=LBFGSB(maxiter=12),
        engineered_features=EngineeredFeatures(
            features=[ProductFeature(key="xx", features=["x", "x"])]
        ),
    )
    assert isinstance(made.kernel, LinearKernel)
    assert isinstance(made.likelihood, SparseOutlierGaussianLikelihood)
    assert made.optimizer.maxiter == 12
    assert made.engineered_features.get_keys() == ["xx"]


def test_make_preserves_runtime_subclass():
    class CustomGP(SingleTaskGPSurrogate):
        pass

    inputs, outputs = features(specs.SingleTaskGPSurrogate)
    assert type(CustomGP.make(inputs, outputs)) is CustomGP


@pytest.mark.parametrize(
    "spec_cls,runtime_cls",
    [
        (specs.PairwiseGPSurrogate, PairwiseGPSurrogate),
        (
            specs.EnsembleMapSaasSingleTaskGPSurrogate,
            EnsembleMapSaasSingleTaskGPSurrogate,
        ),
    ],
)
def test_make_gp_defaults_are_independent(spec_cls, runtime_cls):
    inputs, outputs = features(spec_cls)
    first = runtime_cls.make(inputs, outputs)
    first.optimizer.maxiter = 7
    first.scaler.features.append(inputs.get_keys()[0])
    second = runtime_cls.make(inputs, outputs)
    assert second.optimizer.maxiter is None
    assert second.scaler.features == []
    assert runtime_cls.make(inputs, outputs, scaler=None).scaler is None
