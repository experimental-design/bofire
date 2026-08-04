import pytest

from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import LSRBO, BotorchOptimizer


@pytest.mark.parametrize(
    "gamma, acqf_local, acqf_global, expected",
    [(0.1, 0.3, 0.4, True), (0.4, 0.1, 0.5, False)],
)
def test_LSRBO(gamma, acqf_local, acqf_global, expected):
    assert (
        LSRBO(gamma=gamma).is_local_step(acqf_local=acqf_local, acqf_global=acqf_global)
        == expected
    )


def test_validate_batch_limit():
    acquisition_optimizer = BotorchOptimizer()
    assert acquisition_optimizer.batch_limit == acquisition_optimizer.n_restarts

    acquisition_optimizer = BotorchOptimizer(batch_limit=50)
    assert acquisition_optimizer.batch_limit == acquisition_optimizer.n_restarts
    acquisition_optimizer = BotorchOptimizer(batch_limit=2, n_restarts=4)
    assert acquisition_optimizer.batch_limit == 2
    assert acquisition_optimizer.n_restarts == 4


def _make_outputs():
    return [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]


def test_validate_domain_lsrbo_with_nchoosek_is_rejected():
    """NChooseK is not a `LinearConstraint`, so the existing
    `LSR-BO only supported for linear constraints` check already
    rejects this combination at data-model validation time.
    """
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(3)],
        outputs=_make_outputs(),
        constraints=[
            NChooseKConstraint(
                features=["x1", "x2", "x3"],
                min_count=1,
                max_count=2,
                none_also_valid=False,
            ),
        ],
    )
    optimizer = BotorchOptimizer(local_search_config=LSRBO(gamma=0.1))
    with pytest.raises(ValueError, match="LSR-BO only supported for linear"):
        optimizer.validate_domain(domain)


def test_validate_domain_lsrbo_with_semicontinuous_is_rejected():
    """Semi-continuous features (`allow_zero=True` with `bounds[0] > 0`)
    create a disconnected feasible region that LSR-BO's shortest-path
    interpolation cannot traverse. The validator must reject this
    combination at data-model validation time, with a feature-specific
    message.
    """
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0.2, 1.0), allow_zero=True),
            ContinuousInput(key="x2", bounds=(0.0, 1.0)),
        ],
        outputs=_make_outputs(),
    )
    optimizer = BotorchOptimizer(local_search_config=LSRBO(gamma=0.1))
    with pytest.raises(ValueError, match="semi-continuous"):
        optimizer.validate_domain(domain)


def test_validate_domain_lsrbo_without_pruning_features_passes():
    """LSR-BO with a domain that has neither NChooseK nor semi-continuous
    features (a plain continuous domain) is still supported — the
    validator should NOT raise.
    """
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(3)],
        outputs=_make_outputs(),
    )
    optimizer = BotorchOptimizer(local_search_config=LSRBO(gamma=0.1))
    # Should not raise. A warning about missing local_search_region is
    # acceptable (and expected — see has_local_search_region check).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.validate_domain(domain)
