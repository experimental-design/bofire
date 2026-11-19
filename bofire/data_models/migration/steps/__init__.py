from bofire.data_models.migration.steps.pre_to_0_3_3 import run as run_pre_to_baseline


STEPS = [
    ("pre", "0.3.3", run_pre_to_baseline),
]
