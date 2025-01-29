import random
import uuid

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.constraints.api import (
    ConstraintNotFulfilledError,
    LinearEqualityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Input,
    Output,
)


# TODO: split this into the individual methods in the `Inputs` and `Outputs` classes


if1 = specs.features.valid(ContinuousInput).obj(key="cont")
if2 = specs.features.valid(CategoricalInput).obj(key="cat")
if3 = specs.features.valid(CategoricalDescriptorInput).obj(key="cat_")
if4 = specs.features.valid(CategoricalInput).obj(
    key="cat2",
    allowed=[True, True, False],
)
if5 = specs.features.valid(ContinuousInput).obj(
    key="if5",
    bounds=(3, 3),
)
if6 = specs.features.valid(CategoricalInput).obj(
    key="if6",
    categories=["c1", "c2", "c3"],
    allowed=[True, False, False],
)
of1 = specs.features.valid(ContinuousOutput).obj(key="out1")
of2 = specs.features.valid(ContinuousOutput).obj(key="out2")


def generate_experiments(
    domain: Domain,
    row_count: int = 5,
    tol: float = 0.0,
    only_allowed_categories: bool = False,
    force_all_categories: bool = False,
    include_labcode: bool = False,
):
    if only_allowed_categories:
        assert force_all_categories is False
    if force_all_categories:
        assert only_allowed_categories is False
    experiments = pd.DataFrame.from_dict(
        [
            {
                **{
                    f.key: random.uniform(f.lower_bound - tol, f.upper_bound + tol)
                    for f in domain.inputs.get(ContinuousInput)
                },
                **{
                    f.key: random.choice(f.values)
                    for f in domain.inputs.get(DiscreteInput)
                },
                **{
                    k: random.random()
                    for k in [
                        *domain.outputs.get_keys(ContinuousOutput),
                    ]
                },
                **{
                    f.key: (
                        random.choice(f.categories)
                        if not only_allowed_categories
                        else random.choice(f.get_allowed_categories())
                    )
                    for f in domain.inputs.get(CategoricalInput)
                },
            }
            for _ in range(row_count)
        ],
    )
    if include_labcode:
        experiments["labcode"] = [str(i) for i in range(row_count)]
    if force_all_categories:
        for feat in domain.inputs.get(CategoricalInput):
            categories = (
                feat.categories
                if len(feat.categories) <= row_count
                else feat.categories[:row_count]
            )
            experiments.loc[: len(categories) - 1, feat.key] = categories

    return experiments


def generate_candidates(domain: Domain, row_count: int = 5):
    return pd.DataFrame.from_dict(
        [
            {
                **{
                    feat.key: random.uniform(feat.lower_bound, feat.upper_bound)
                    for feat in domain.inputs.get(ContinuousInput)
                },
                **{
                    f"{k}_pred": random.random()
                    for k in domain.outputs.get_keys(ContinuousOutput)
                },
                **{
                    f"{k}_sd": random.random()
                    for k in domain.outputs.get_keys(ContinuousOutput)
                },
                **{
                    f"{k}_des": random.random()
                    for k in domain.outputs.get_keys(ContinuousOutput)
                },
                **{
                    f.key: random.choice(f.get_allowed_categories())
                    for f in domain.inputs.get(CategoricalInput)
                },
            }
            for _ in range(row_count)
        ],
    )


def generate_invalid_candidates_bounds(domain, row_count: int = 5, error="lower"):
    candidates = generate_candidates(domain, row_count)
    if error == "lower":
        candidates.loc[0, domain.inputs.get_keys(ContinuousInput)[0]] = (
            domain.inputs.get(ContinuousInput)[0].lower_bound - 0.1
        )
    else:
        candidates.loc[0, domain.inputs.get_keys(ContinuousInput)[0]] = (
            domain.inputs.get(ContinuousInput)[0].upper_bound + 0.1
        )
    return candidates


domain0 = Domain.from_lists(
    inputs=[if1, if2, if3],
    outputs=[of1, of2],
    constraints=[],
)

domain1 = Domain.from_lists(
    inputs=[if1, if2],
    outputs=[of1],
    constraints=[],
)

domain2 = Domain.from_lists(
    inputs=[if1, if2, if3],
    outputs=[of1],
    constraints=[],
)

domain3 = Domain.from_lists(
    [if1, if2],
    [of1, of2],
    [],
)

domain4 = Domain.from_lists(
    [if1, if2, if3, if4],
    [of1, of2],
    [],
)
domain5 = Domain.from_lists(
    [if1, if5],
    [of1, of2],
    [],
)
domain6 = Domain.from_lists(
    [if1, if6],
    [of1, of2],
)
domain7 = Domain.from_lists(
    [if1, if5],
    [of1, of2],
    constraints=[
        LinearEqualityConstraint(
            features=["cont", "if5"], coefficients=[1, 1], rhs=500
        ),
    ],
)

domains = [domain0, domain1, domain2, domain3, domain4]


@pytest.mark.parametrize(
    "domain, experiments",
    [
        (d, generate_experiments(d, include_labcode=include_labcode))
        for d in domains + [domain5, domain6]
        for include_labcode in [True, False]
        # (d, generate_experiments(d))
        # for d in domains+[domain5,domain6]
    ],
)
def test_domain_validate_experiments_valid(
    domain: Domain,
    experiments: pd.DataFrame,
):
    experiments1 = domain.validate_experiments(experiments.copy())
    for col in experiments.columns:
        experiments[col] = experiments[col].map(str)
    experiments2 = domain.validate_experiments(experiments)
    assert_frame_equal(
        left=experiments1,
        right=experiments2,
    )


@pytest.mark.parametrize(
    "domain, experiments, strict",
    [
        (domains[0], generate_experiments(domains[1]), True),
        (domains[0], generate_experiments(domains[1]), False),
    ],
)
def test_domain_validate_experiments_invalid(
    domain: Domain,
    experiments: pd.DataFrame,
    strict: bool,
):
    with pytest.raises(ValueError):
        domain.validate_experiments(experiments, strict=strict)


@pytest.mark.parametrize(
    "domain, experiments",
    [
        (d, generate_experiments(d, only_allowed_categories=True))
        for d in [domain5, domain6]
    ],
)
def test_domain_validate_experiments_strict_invalid(
    domain: Domain,
    experiments: pd.DataFrame,
):
    domain.validate_experiments(experiments, strict=False)
    with pytest.raises(ValueError):
        domain.validate_experiments(experiments, strict=True)


def test_domain_validate_experiments_invalid_labcode():
    # non distinct labcodes
    experiments = generate_experiments(domain4, row_count=2, include_labcode=True)
    experiments.labcode = ["1", "1"]
    with pytest.raises(ValueError):
        domain4.validate_experiments(experiments)
    # na labcodes
    experiments = generate_experiments(domain4, row_count=2, include_labcode=True)
    experiments.labcode = [np.nan, "1"]
    with pytest.raises(ValueError):
        domain4.validate_experiments(experiments)


@pytest.mark.parametrize(
    "domain, candidates",
    [(d, generate_candidates(d)) for d in domains],
)
def test_domain_validate_candidates_valid(
    domain: Domain,
    candidates: pd.DataFrame,
):
    candidates1 = domain.validate_candidates(candidates.copy())
    for col in domain.inputs.get_keys():
        candidates[col] = candidates[col].map(str)
    candidates2 = domain.validate_candidates(candidates.copy())
    assert_frame_equal(
        left=candidates1,
        right=candidates2,
    )


@pytest.mark.parametrize(
    "domain, candidates",
    [
        (d, generate_candidates(d).drop(key, axis=1))
        for d in [domain0]
        for key in d.inputs.get_keys(Input)
    ]
    + [
        (d, generate_candidates(d).drop(key, axis=1))
        for d in [domain0]
        for key_ in d.outputs.get_keys(Output)
        for key in [f"{key_}_pred", f"{key_}_sd", f"{key_}_des"]
    ],
)
def test_domain_validate_candidates_missing_cols(
    domain: Domain,
    candidates: pd.DataFrame,
):
    with pytest.raises(ValueError):
        domain.validate_candidates(candidates)


@pytest.mark.parametrize(
    "domain, candidates",
    [
        (domain0, generate_invalid_candidates_bounds(domain0, error="lower")),
        (domain0, generate_invalid_candidates_bounds(domain0, error="upper")),
    ],
)
def test_domain_validate_candidates_invalid_bounds(
    domain: Domain,
    candidates: pd.DataFrame,
):
    with pytest.raises(ValueError):
        domain.validate_candidates(candidates)


def test_domain_validate_candidates_invalid_categories():
    candidates = generate_candidates(domain4)
    candidates.loc[0, "cat2"] = "c3"
    with pytest.raises(ValueError):
        domain4.validate_candidates(candidates)


@pytest.mark.parametrize(
    "domain, candidates, key",
    [
        (d, generate_candidates(d), key)
        for d in domains
        for key in d.inputs.get_keys(Input)
    ]
    + [
        (d, generate_candidates(d), key)
        for d in domains
        for key_ in d.outputs.get_keys(ContinuousOutput)
        for key in [f"{key_}_pred", f"{key_}_sd", f"{key_}_des"]
    ],
)
def test_domain_validate_candidates_not_numerical(
    domain: Domain,
    candidates: pd.DataFrame,
    key: str,
):
    candidates[key] = str(uuid.uuid4())
    with pytest.raises(ValueError):
        domain.validate_candidates(candidates)


@pytest.mark.parametrize(
    "domain, candidates, raise_validation_error",
    [
        (d, generate_candidates(d), raise_validation_error)
        for d in [domain7]
        for raise_validation_error in [True, False]
    ],
)
def test_domain_validate_candidates_constraint_not_fulfilled(
    domain,
    candidates,
    raise_validation_error,
):
    if raise_validation_error:
        with pytest.raises(ConstraintNotFulfilledError):
            domain.validate_candidates(
                candidates,
                raise_validation_error=raise_validation_error,
            )
    else:
        assert isinstance(
            domain.validate_candidates(
                candidates,
                raise_validation_error=raise_validation_error,
            ),
            pd.DataFrame,
        )


def test_outputs_add_valid_columns():
    experiments = generate_experiments(domain=domain0)
    assert "valid_out1" not in experiments.columns
    assert "valid_out2" not in experiments.columns
    experiments = domain0.outputs.add_valid_columns(experiments)
    assert "valid_out1" in experiments.columns
    assert "valid_out2" in experiments.columns
    experiments["valid_out1"] = "1"
    experiments["valid_out2"] = "0"
    experiments = domain0.outputs.add_valid_columns(experiments)
    assert (experiments["valid_out1"] == 1).all()
    assert (experiments["valid_out2"] == 0).all()
    experiments["valid_out1"] = 0
    experiments["valid_out2"] = 1
    experiments = domain0.outputs.add_valid_columns(experiments)
    assert (experiments["valid_out1"] == 0).all()
    assert (experiments["valid_out2"] == 1).all()
    experiments["valid_out1"] = 0.0
    experiments["valid_out2"] = 1.0
    experiments = domain0.outputs.add_valid_columns(experiments)
    assert (experiments["valid_out1"] == 0).all()
    assert (experiments["valid_out2"] == 1).all()
    for _test_val in ["1.0", "1.2"]:
        experiments["valid_out1"] = _test_val
        with pytest.raises(ValueError):
            domain0.outputs.add_valid_columns(experiments)
