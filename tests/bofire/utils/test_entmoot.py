
from entmoot.problem_config import FeatureType, ProblemConfig

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.utils.entmoot import domain_to_problem_config


def feat_equal(a: FeatureType, b: FeatureType) -> bool:
    """Check if entmoot.FeatureTypes are equal.

    Args:
        a: First feature.
        b: Second feature.
    """
    # no __eq__ method is implemented for FeatureType, hence the need for this function

    return all(
        (
            a.name == b.name,
            a.get_enc_bnds() == b.get_enc_bnds(),
            a.is_real() == b.is_real(),
            a.is_cat() == b.is_cat(),
            a.is_int() == b.is_int(),
            a.is_bin() == b.is_bin(),
        )
    )


if1 = CategoricalInput(key="if1", categories=("blue", "orange", "gray"))
if1_ent = {
    "feat_type": "categorical",
    "bounds": ("blue", "orange", "gray"),
    "name": "if1",
}

if2 = DiscreteInput(key="if2", values=[5, 6, 7])
if2_ent = {"feat_type": "integer", "bounds": (5, 7), "name": "if2"}

if3 = DiscreteInput(key="if3", values=[0, 1])
if3_ent = {"feat_type": "binary", "name": "if3"}

if4 = ContinuousInput(key="if4", bounds=[5.0, 6.0])
if4_ent = {"feat_type": "real", "bounds": (5.0, 6.0), "name": "if4"}

if5 = ContinuousInput(key="if5", bounds=[0.0, 10.0])

of1 = ContinuousOutput(key="of1", objective=MinimizeObjective(w=1.0))
of1_ent = {"name": "of1"}

of2 = ContinuousOutput(key="of2", objective=MaximizeObjective(w=1.0))
of2_ent = {"name": "of2"}

constr1 = LinearInequalityConstraint(
    features=["if4", "if5"], coefficients=[1, 1], rhs=12
)
constr2 = LinearEqualityConstraint(features=["if4", "if5"], coefficients=[1, 5], rhs=38)


def build_problem_config(inputs, outputs) -> ProblemConfig:
    problem_config = ProblemConfig()
    for feature in inputs:
        problem_config.add_feature(**feature)

    for objective in outputs:
        problem_config.add_min_objective(**objective)

    return problem_config


def test_domain_to_problem_config():
    domain = Domain.from_lists(inputs=[if1, if2, if3, if4], outputs=[of1, of2])
    ent_problem_config = build_problem_config(
        inputs=[if1_ent, if2_ent, if3_ent, if4_ent], outputs=[of1_ent, of2_ent]
    )
    bof_problem_config, _ = domain_to_problem_config(domain)

    for feat_a, feat_b in zip(
        ent_problem_config.feat_list, bof_problem_config.feat_list
    ):
        assert feat_equal(feat_a, feat_b)

    assert len(ent_problem_config.obj_list) == len(bof_problem_config.obj_list)


def test_convert_constraint_to_entmoot():
    constraints = [constr1, constr2]
    domain = Domain.from_lists(
        inputs=[if1, if2, if3, if4, if5], outputs=[of1, of2], constraints=constraints
    )
    _, model = domain_to_problem_config(domain)

    assert len(constraints) == len(model.constr)
