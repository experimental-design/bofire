from bofire.data_models.migration.errors import UnrecoverablePayloadError
from bofire.data_models.migration.registry import normalizer
from bofire.data_models.migration.steps.pre_to_0_3_3._atoms import drop_keys


STEP = "pre_to_0_3_3"

_SOBO_OBSOLETE = (
    "categorical_method",
    "descriptor_method",
    "discrete_method",
    "num_raw_samples",
    "num_restarts",
)


@normalizer(STEP, "BotorchOptimizer")
def normalize_botorch_optimizer(p: dict) -> dict:
    """Drop legacy categorical_/descriptor_/discrete_method fields.

    Some MoboStrategy records stored these fields on the ``acquisition_optimizer``
    (BotorchOptimizer) rather than at the strategy top level. Current
    BotorchOptimizer has no such fields.
    """
    return drop_keys(p, _SOBO_OBSOLETE)


@normalizer(STEP, "SoboStrategy", "MoboStrategy", "MultiplicativeSoboStrategy")
def normalize_sobo_family(p: dict) -> dict:
    return drop_keys(p, _SOBO_OBSOLETE)


@normalizer(STEP, "QnehviStrategy")
def migrate_qnehvi_to_mobo(p: dict) -> dict:
    """QnehviStrategy was deprecated; rewrite as MoboStrategy.

    MoboStrategy with its default acquisition_function=qLogNEHVI() subsumes the
    qNEHVI use case. All shared fields (ref_point, surrogate_specs,
    outlier_detection_specs, frequency_check, frequency_hyperopt, folds,
    min_experiments_before_outlier_check, seed) carry over unchanged.
    """
    drop_keys(p, ["alpha", *_SOBO_OBSOLETE])
    p["type"] = "MoboStrategy"
    return p


@normalizer(STEP, "CustomSoboStrategy")
def reject_custom_sobo(p: dict) -> dict:
    raise UnrecoverablePayloadError(
        payload_type="CustomSoboStrategy",
        reason="CustomSoboStrategy is not supported by the migration tool.",
        hint="Recreate the experiment as a SoboStrategy with appropriate config.",
    )


# Strategies that need no field migration; the walker still recurses into
# their children. Registering identity ensures discriminator coverage.
@normalizer(
    STEP,
    "RandomStrategy",
    "QparegoStrategy",
    "FractionalFactorialStrategy",
    "DoEStrategy",
    "StepwiseStrategy",
)
def passthrough(p: dict) -> dict:
    return p
