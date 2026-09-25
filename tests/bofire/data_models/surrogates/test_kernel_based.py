"""Surrogates check their kernel, mean and likelihood against the features at construction."""

import pytest
from pydantic import ValidationError

import bofire.data_models.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.encodings.api import OneHotEncoding, OrdinalEncoding
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    HammingDistanceKernel,
    ICMKernel,
    IndexKernel,
    LinearKernel,
    MixedKernel,
    RBFKernel,
    ScaleKernel,
)
from bofire.data_models.likelihoods.api import TaskGaussianLikelihood
from bofire.data_models.means.api import TaskConstantMean
from bofire.data_models.surrogates.kernel_based import KernelBasedSurrogate


INPUTS = Inputs(
    features=[
        ContinuousInput(key="x", bounds=(0, 1)),
        CategoricalInput(key="c", categories=["a", "b"]),
        CategoricalInput(key="d", categories=["p", "q", "r"]),
    ]
)
OUTPUTS = Outputs(features=[ContinuousOutput(key="y")])
ORDINAL = {"c": OrdinalEncoding(), "d": OrdinalEncoding()}


def _gp(kernel, encodings=None):
    return surrogates.SingleTaskGPSurrogate(
        inputs=INPUTS,
        outputs=OUTPUTS,
        kernel=kernel,
        categorical_encodings=encodings or ORDINAL,
    )


@pytest.mark.parametrize(
    "kernel, match",
    [
        pytest.param(
            HammingDistanceKernel(features=["x"]),
            r"HammingDistanceKernel cannot work on \['x'\]",
            id="categorical-kernel-on-continuous",
        ),
        pytest.param(
            ScaleKernel(base_kernel=HammingDistanceKernel(features=["x"])),
            r"HammingDistanceKernel cannot work on \['x'\]",
            id="inside-scale",
        ),
        pytest.param(
            AdditiveKernel(
                kernels=[
                    RBFKernel(features=["x"]),
                    ScaleKernel(base_kernel=HammingDistanceKernel(features=["x"])),
                ]
            ),
            r"HammingDistanceKernel cannot work on \['x'\]",
            id="nested-in-additive",
        ),
        pytest.param(
            RBFKernel(features=["nope"]),
            r"names \['nope'\], which are neither inputs nor engineered features",
            id="unknown-key",
        ),
        pytest.param(
            IndexKernel(num_categories=2, features=["c", "d"]),
            "works on exactly one feature",
            id="index-kernel-over-two-features",
        ),
        pytest.param(
            HammingDistanceKernel(),
            r"HammingDistanceKernel cannot work on \['x'\]",
            id="unset-features-offer-everything",
        ),
    ],
)
def test_single_task_gp_rejects_kernels_that_cannot_work(kernel, match):
    with pytest.raises(ValidationError, match=match):
        _gp(kernel)


@pytest.mark.parametrize(
    "kernel, encodings",
    [
        pytest.param(RBFKernel(), ORDINAL, id="rbf-on-ordinal-codes"),
        pytest.param(
            HammingDistanceKernel(features=["c"]),
            {"c": OneHotEncoding(), "d": OneHotEncoding()},
            id="hamming-on-one-hot",
        ),
        pytest.param(
            AdditiveKernel(
                kernels=[
                    RBFKernel(features=["x"]),
                    HammingDistanceKernel(features=["c", "d"]),
                ]
            ),
            ORDINAL,
            id="routed-by-hand",
        ),
    ],
)
def test_single_task_gp_accepts_kernels_that_work(kernel, encodings):
    _gp(kernel, encodings)


def test_assigning_a_kernel_is_validated():
    """Hyperparameter search assigns kernels, and `validate_assignment` reruns the check."""
    surrogate = _gp(RBFKernel())

    with pytest.raises(ValidationError, match="cannot work on"):
        surrogate.kernel = HammingDistanceKernel(features=["x"])


def test_validation_does_not_modify_components():
    kernel = ScaleKernel(base_kernel=RBFKernel())
    before = kernel.model_dump()

    surrogate = _gp(kernel)

    assert surrogate.kernel.model_dump() == before
    assert surrogate.kernel.base_kernel.features is None


@pytest.mark.parametrize(
    "cls, validated",
    [
        (surrogates.SingleTaskGPSurrogate, True),
        (surrogates.RobustSingleTaskGPSurrogate, True),
        (surrogates.TanimotoGPSurrogate, True),
        (surrogates.PairwiseGPSurrogate, True),
        (surrogates.LinearSurrogate, True),
        (surrogates.PolynomialSurrogate, True),
        (surrogates.MultiTaskGPSurrogate, True),
        (surrogates.MixedSingleTaskGPSurrogate, True),
    ],
)
def test_which_surrogates_validate_their_components(cls, validated):
    assert issubclass(cls, KernelBasedSurrogate) is validated


def test_linear_surrogate_rejects_an_unknown_key():
    with pytest.raises(ValidationError, match=r"names \['nope'\]"):
        surrogates.LinearSurrogate(
            inputs=INPUTS,
            outputs=OUTPUTS,
            kernel=LinearKernel(features=["nope"]),
        )


TASK_ORDINAL = {"t": OrdinalEncoding()}
TASK_INPUTS = Inputs(
    features=[
        ContinuousInput(key="x", bounds=(0, 1)),
        CategoricalTaskInput(key="t", categories=["a", "b", "c"]),
    ]
)


@pytest.mark.parametrize(
    "inputs, kwargs, match",
    [
        pytest.param(
            Inputs(features=[ContinuousInput(key="x", bounds=(0, 1))]),
            {"kernel": ICMKernel(base_kernel=RBFKernel())},
            "Exactly one task input is required",
            id="icm-without-task-input",
        ),
        pytest.param(
            TASK_INPUTS,
            {
                "kernel": ICMKernel(base_kernel=RBFKernel()),
                "categorical_encodings": {"t": OneHotEncoding()},
            },
            "has to be encoded as ordinal codes",
            id="icm-on-one-hot-task",
        ),
        pytest.param(
            TASK_INPUTS,
            {
                "kernel": ICMKernel(base_kernel=RBFKernel(), rank=4),
                "categorical_encodings": TASK_ORDINAL,
            },
            "rank=4, but 't' has only 3 tasks",
            id="icm-rank-above-task-count",
        ),
        pytest.param(
            Inputs(features=[ContinuousInput(key="x", bounds=(0, 1))]),
            {"mean": TaskConstantMean()},
            "Exactly one task input is required",
            id="task-mean-without-task-input",
        ),
        pytest.param(
            Inputs(features=[ContinuousInput(key="x", bounds=(0, 1))]),
            {"likelihood": TaskGaussianLikelihood()},
            "Exactly one task input is required",
            id="task-likelihood-without-task-input",
        ),
    ],
)
def test_task_components_reject_unusable_task_inputs(inputs, kwargs, match):
    with pytest.raises(ValidationError, match=match):
        surrogates.SingleTaskGPSurrogate(inputs=inputs, outputs=OUTPUTS, **kwargs)


def test_task_components_accept_a_task_input():
    surrogates.SingleTaskGPSurrogate(
        inputs=TASK_INPUTS,
        outputs=OUTPUTS,
        kernel=ICMKernel(base_kernel=RBFKernel(), rank=2),
        mean=TaskConstantMean(),
        likelihood=TaskGaussianLikelihood(),
        # a single-task GP one-hot encodes task inputs by default; the task components
        # need the ordinal codes
        categorical_encodings=TASK_ORDINAL,
    )


def test_mixed_gp_honours_explicit_features_on_its_kernels():
    """A continuous kernel may act on ordinal codes if asked to; the split is a default."""
    surrogate = surrogates.MixedSingleTaskGPSurrogate(
        inputs=INPUTS,
        outputs=OUTPUTS,
        continuous_kernel=RBFKernel(features=["x", "c"]),
    )

    assert surrogate.continuous_kernel.features == ["x", "c"]


MIXED_INPUTS = Inputs(
    features=[
        ContinuousInput(key="x", bounds=(0, 1)),
        CategoricalInput(key="c", categories=["a", "b"]),
        CategoricalTaskInput(key="t", categories=["u", "v"]),
    ]
)


def _encodings(surrogate):
    return {k: type(v).__name__ for k, v in surrogate.categorical_encodings.items()}


def test_components_request_the_encoding_they_need():
    mixed = surrogates.SingleTaskGPSurrogate(
        inputs=MIXED_INPUTS, outputs=OUTPUTS, kernel=MixedKernel()
    )
    task = surrogates.SingleTaskGPSurrogate(
        inputs=MIXED_INPUTS,
        outputs=OUTPUTS,
        kernel=ICMKernel(base_kernel=RBFKernel()),
        mean=TaskConstantMean(),
    )

    # MixedKernel asks for ordinal codes on every plain categorical
    assert _encodings(mixed) == {"c": "OrdinalEncoding", "t": "OrdinalEncoding"}
    # the task components only on the task input; `c` keeps the one-hot default
    assert _encodings(task) == {"c": "OneHotEncoding", "t": "OrdinalEncoding"}


def test_an_explicit_encoding_wins_over_a_request():
    with pytest.raises(ValidationError, match="has to be encoded as ordinal codes"):
        surrogates.SingleTaskGPSurrogate(
            inputs=MIXED_INPUTS,
            outputs=OUTPUTS,
            kernel=ICMKernel(base_kernel=RBFKernel()),
            categorical_encodings={"t": OneHotEncoding()},
        )


class _Requesting:
    """A component asking for one encoding of one feature."""

    def __init__(self, encoding):
        self.encoding = encoding

    def encoding_requests(self, inputs):
        return {"c": self.encoding}

    def validate_inputs(self, context):
        pass


def test_conflicting_requests_raise():
    surrogate = surrogates.SingleTaskGPSurrogate(inputs=MIXED_INPUTS, outputs=OUTPUTS)
    surrogate.__dict__["components"] = lambda: [
        _Requesting(OrdinalEncoding()),
        _Requesting(OneHotEncoding()),
    ]

    with pytest.raises(ValueError, match="ask for different encodings of 'c'"):
        surrogate.encoding_requests()


def test_default_encodings_do_not_depend_on_the_components_mixin():
    """Without requests, a surrogate with components fills in what one without does."""
    with_mixin = surrogates.SingleTaskGPSurrogate(inputs=MIXED_INPUTS, outputs=OUTPUTS)
    without = surrogates.RandomForestSurrogate(inputs=MIXED_INPUTS, outputs=OUTPUTS)

    assert _encodings(with_mixin) == _encodings(without)


def test_requested_encodings_survive_a_round_trip():
    surrogate = surrogates.SingleTaskGPSurrogate(
        inputs=MIXED_INPUTS, outputs=OUTPUTS, kernel=MixedKernel()
    )

    restored = surrogates.SingleTaskGPSurrogate.model_validate_json(
        surrogate.model_dump_json()
    )

    assert restored.categorical_encodings == surrogate.categorical_encodings
    assert restored.kernel == surrogate.kernel
