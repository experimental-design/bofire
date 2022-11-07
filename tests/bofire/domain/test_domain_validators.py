import random
import uuid
from typing import List

import numpy as np
import pandas as pd
import pytest

from bofire.domain.domain import Domain
from bofire.domain.features import (
    CategoricalDescriptorInputFeature,
    CategoricalInputFeature,
    ContinuousInputFeature,
    ContinuousOutputFeature,
    InputFeature,
    OutputFeature,
)
from tests.bofire.domain.test_features import (
    VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
)

if1 = ContinuousInputFeature(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "cont",
    }
)
if2 = CategoricalInputFeature(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "cat",
    }
)
if3 = CategoricalDescriptorInputFeature(
    **{
        **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "cat_",
    }
)
if4 = CategoricalInputFeature(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "cat2",
        "allowed": [True, True, False],
    }
)
if5 = ContinuousInputFeature(
    **{
        **VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if5",
    }
)
if6 = CategoricalInputFeature(
    **{
        **VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if6",
    }
)
of1 = ContinuousOutputFeature(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "out1",
    }
)
of2 = ContinuousOutputFeature(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "out2",
    }
)


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
                    for f in domain.get_features(ContinuousInputFeature)
                },
                **{
                    k: random.random()
                    for k in [
                        *domain.get_feature_keys(ContinuousOutputFeature),
                    ]
                },
                **{
                    f.key: random.choice(f.categories)
                    if not only_allowed_categories
                    else random.choice(f.get_allowed_categories())
                    for f in domain.get_features(CategoricalInputFeature)
                },
            }
            for _ in range(row_count)
        ]
    )
    if include_labcode:
        experiments["labcode"] = [str(i) for i in range(row_count)]
    if force_all_categories:
        for feat in domain.get_features(CategoricalInputFeature):
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
                    for feat in domain.get_features(ContinuousInputFeature)
                },
                **{
                    f"{k}_pred": random.random()
                    for k in domain.get_feature_keys(ContinuousOutputFeature)
                },
                **{
                    f"{k}_sd": random.random()
                    for k in domain.get_feature_keys(ContinuousOutputFeature)
                },
                **{
                    f"{k}_des": random.random()
                    for k in domain.get_feature_keys(ContinuousOutputFeature)
                },
                **{
                    f.key: random.choice(f.get_allowed_categories())
                    for f in domain.get_features(CategoricalInputFeature)
                },
            }
            for _ in range(row_count)
        ]
    )


def generate_invalid_candidates_bounds(domain, row_count: int = 5, error="lower"):
    candidates = generate_candidates(domain, row_count)
    if error == "lower":
        candidates.loc[0, domain.get_feature_keys(ContinuousInputFeature)[0]] = (
            domain.get_features(ContinuousInputFeature)[0].lower_bound - 0.1
        )
    else:
        candidates.loc[0, domain.get_feature_keys(ContinuousInputFeature)[0]] = (
            domain.get_features(ContinuousInputFeature)[0].upper_bound + 0.1
        )
    return candidates


domain0 = Domain(
    input_features=[if1, if2, if3],
    output_features=[of1, of2],
    constraints=[],
)

domain1 = Domain(
    input_features=[if1, if2],
    output_features=[of1],
    constraints=[],
)

domain2 = Domain(
    input_features=[if1, if2, if3],
    output_features=[of1],
    constraints=[],
)

domain3 = Domain(
    input_features=[if1, if2],
    output_features=[of1, of2],
    constraints=[],
)

domain4 = Domain(
    input_features=[if1, if2, if3, if4],
    output_features=[of1, of2],
    constraints=[],
)
domain5 = Domain(
    input_features=[if1, if5],
    output_features=[of1, of2],
    constraints=[],
)
domain6 = Domain(
    input_features=[if1, if6],
    output_features=[of1, of2],
    constraints=[],
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
    domain.validate_experiments(experiments)


@pytest.mark.parametrize(
    "domain, experiments, strict",
    [
        (d1, generate_experiments(d2), strict)
        for strict in [True, False]
        for d1 in domains
        for d2 in domains
        if d1 != d2
    ],
)
def test_domain_validate_experiments_invalid(
    domain: Domain, experiments: pd.DataFrame, strict: bool
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
    "domain, candidates", [(d, generate_candidates(d)) for d in domains]
)
def test_domain_validate_candidates_valid(
    domain: Domain,
    candidates: pd.DataFrame,
):
    domain.validate_candidates(candidates)


@pytest.mark.parametrize(
    "domain, candidates",
    [
        (d, generate_candidates(d).drop(key, axis=1))
        for d in [domain0]
        for key in d.get_feature_keys(InputFeature)
    ]
    + [
        (d, generate_candidates(d).drop(key, axis=1))
        for d in [domain0]
        for key_ in d.get_feature_keys(OutputFeature)
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
    domain: Domain, candidates: pd.DataFrame
):
    with pytest.raises(ValueError):
        domain.validate_candidates(candidates)


def test_domain_validate_candidates_invalid_categories():
    candidates = generate_candidates(domain4)
    candidates.loc[0, "cat2"] = "c3"
    with pytest.raises(ValueError):
        domain4.validate_candidates(candidates)


@pytest.mark.parametrize(
    "domain, candidates, cols",
    [
        (d, generate_candidates(d), cols)
        for d in [domain0]
        for cols in [
            ["newcol"],
            ["newcol1", "newcol2"],
        ]
    ],
)
def test_domain_validate_candidates_too_many_cols(
    domain: Domain,
    candidates: pd.DataFrame,
    cols: List[str],
):
    candidates = candidates.reindex(list(candidates.columns) + cols, axis=1)
    with pytest.raises(ValueError):
        domain.validate_candidates(candidates)


@pytest.mark.parametrize(
    "domain, candidates, key",
    [
        (d, generate_candidates(d), key)
        for d in domains
        for key in d.get_feature_keys(InputFeature)
    ]
    + [
        (d, generate_candidates(d), key)
        for d in domains
        for key_ in d.get_feature_keys(ContinuousOutputFeature)
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
