from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.predictives.acqf_optimization import (
    BotorchOptimizer as BotorchOptimizerModel,
)
from bofire.strategies.predictives.acqf_optimization import BotorchOptimizer


def test_determine_optimizer():
    optimizer_data = BotorchOptimizerModel()
    domain = Domain(
        inputs=[
            ContinuousInput(name="x1", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(name="y1")],
    )
    optimizer = BotorchOptimizer(optimizer_data)
    assert optimizer.determine_optimizer(domain, n_acqfs=2) == "optimize_acqf_list"
    assert optimizer.determine_optimizer(domain, n_acqfs=1) == "optimize_acqf"
    domain = Domain(
        inputs=[
            ContinuousInput(name="x1", bounds=(0, 1)),
            CategoricalInput(name="x2", categories=["a", "b"]),
        ],
        outputs=[ContinuousOutput(name="y1")],
    )
    assert optimizer.determine_optimizer(domain, n_acqfs=1) == "optimize_mixed"
    domain = Domain(
        inputs=[
            ContinuousInput(name="x1", bounds=(0, 1)),
            CategoricalInput(name="x2", categories=[f"cat_{i}" for i in range(12)]),
        ],
        outputs=[ContinuousOutput(name="y1")],
    )
    assert (
        optimizer.determine_optimizer(domain, n_acqfs=1) == "optimize_mixed_alternating"
    )
    domain = Domain(
        inputs=[
            ContinuousInput(name="x1", bounds=(0, 1)),
            ContinuousInput(name="x2", bounds=(0, 1)),
            CategoricalInput(name="x3", categories=[f"cat_{i}" for i in range(12)]),
        ],
        constraints=[
            NChooseKConstraint(features=["x1", "x2"], min_count=0, max_count=1)
        ],
        outputs=[ContinuousOutput(name="y1")],
    )
    assert optimizer.determine_optimizer(domain, n_acqfs=1) == "optimize_mixed"


# def test_get_arguments_for_optimizer():
