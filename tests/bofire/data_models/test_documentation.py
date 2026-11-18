"""Documentation ratchet for the data models.

BoFire is configured through its data models, which makes them the API surface for
both humans and agents. The prose therefore belongs where `model_json_schema()` can
see it: `Field(description=...)` for every field, and a docstring on every class.

The allowlists below record what predates that convention. They may only shrink --
`test_field_allowlist_has_no_stale_entries` and its class-level counterpart fail once
an allowlisted entry has been documented (or removed) without being deleted here, so
documenting a package and pruning its entries have to happen in the same commit.
"""

import importlib
import pkgutil
from typing import Dict, List, Type

import pytest
from pydantic import BaseModel

import bofire.data_models


# Fields that do not yet carry `Field(description=...)`. Only ever remove entries.
UNDOCUMENTED_FIELDS = {
    "bofire.data_models.strategies.convergence_criteria.hypervolume_improvement.HypervolumeImprovementCriterion.min_improvement",
    "bofire.data_models.strategies.convergence_criteria.hypervolume_improvement.HypervolumeImprovementCriterion.n_lookback",
    "bofire.data_models.strategies.convergence_criteria.objective_improvement.ObjectiveImprovementCriterion.min_improvement",
    "bofire.data_models.strategies.convergence_criteria.objective_improvement.ObjectiveImprovementCriterion.n_lookback",
    "bofire.data_models.strategies.convergence_criteria.proposal_deviation.ProposalDeviationCriterion.n_consecutive",
    "bofire.data_models.strategies.convergence_criteria.proposal_deviation.ProposalDeviationCriterion.threshold",
    "bofire.data_models.strategies.doe.DoEOptimalityCriterion.formula",
    "bofire.data_models.strategies.doe.DoEStrategy.criterion",
    "bofire.data_models.strategies.doe.DoEStrategy.ipopt_options",
    "bofire.data_models.strategies.doe.DoEStrategy.return_fixed_candidates",
    "bofire.data_models.strategies.doe.DoEStrategy.sampling",
    "bofire.data_models.strategies.doe.DoEStrategy.scip_params",
    "bofire.data_models.strategies.doe.DoEStrategy.use_cyipopt",
    "bofire.data_models.strategies.doe.DoEStrategy.use_hessian",
    "bofire.data_models.strategies.doe.DoEStrategy.verbose",
    "bofire.data_models.strategies.doe.IOptimalityCriterion.ipopt_options",
    "bofire.data_models.strategies.doe.IOptimalityCriterion.n_space_filling_points",
    "bofire.data_models.strategies.doe.OptimalityCriterion.delta",
    "bofire.data_models.strategies.doe.OptimalityCriterion.transform_range",
    "bofire.data_models.strategies.doe.SpaceFillingCriterion.sampling_fraction",
    "bofire.data_models.strategies.fractional_factorial.FractionalFactorialStrategy.block_feature_key",
    "bofire.data_models.strategies.llm.LLMStrategy.llm",
    "bofire.data_models.strategies.llm.LLMStrategy.model_settings",
    "bofire.data_models.strategies.llm.LLMStrategy.n_recent_experiments",
    "bofire.data_models.strategies.llm.LLMStrategy.n_top_experiments",
    "bofire.data_models.strategies.llm.LLMStrategy.output_retries",
    "bofire.data_models.strategies.llm.LLMStrategy.system_prompt",
    "bofire.data_models.strategies.predictives.acqf_optimization.AcquisitionOptimizer.prefer_exhaustive_search_for_purely_categorical_domains",
    "bofire.data_models.strategies.predictives.acqf_optimization.BotorchOptimizer.batch_limit",
    "bofire.data_models.strategies.predictives.acqf_optimization.BotorchOptimizer.local_search_config",
    "bofire.data_models.strategies.predictives.acqf_optimization.BotorchOptimizer.maxiter",
    "bofire.data_models.strategies.predictives.acqf_optimization.BotorchOptimizer.n_raw_samples",
    "bofire.data_models.strategies.predictives.acqf_optimization.BotorchOptimizer.n_restarts",
    "bofire.data_models.strategies.predictives.acqf_optimization.BotorchOptimizer.sequential",
    "bofire.data_models.strategies.predictives.acqf_optimization.GeneticAlgorithmOptimizer.cvtol",
    "bofire.data_models.strategies.predictives.acqf_optimization.GeneticAlgorithmOptimizer.ftol",
    "bofire.data_models.strategies.predictives.acqf_optimization.GeneticAlgorithmOptimizer.n_max_evals",
    "bofire.data_models.strategies.predictives.acqf_optimization.GeneticAlgorithmOptimizer.n_max_gen",
    "bofire.data_models.strategies.predictives.acqf_optimization.GeneticAlgorithmOptimizer.population_size",
    "bofire.data_models.strategies.predictives.acqf_optimization.GeneticAlgorithmOptimizer.verbose",
    "bofire.data_models.strategies.predictives.acqf_optimization.GeneticAlgorithmOptimizer.xtol",
    "bofire.data_models.strategies.predictives.acqf_optimization.LSRBO.gamma",
    "bofire.data_models.strategies.predictives.active_learning.ActiveLearningStrategy.acquisition_function",
    "bofire.data_models.strategies.predictives.botorch.BotorchStrategy.acquisition_optimizer",
    "bofire.data_models.strategies.predictives.botorch.BotorchStrategy.surrogate_specs",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.acq_sense",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.beta",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.bound_coeff",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.cat_metric",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.dist_metric",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.dist_trafo",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.kappa_fantasy",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.max_depth",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.min_data_in_leaf",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.min_data_per_group",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.num_boost_round",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.solver_name",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.solver_params",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.solver_verbose",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy.verbose",
    "bofire.data_models.strategies.predictives.mobo.AbsoluteMovingReferenceValue.offset",
    "bofire.data_models.strategies.predictives.mobo.AbsoluteMovingReferenceValue.orient_at_best",
    "bofire.data_models.strategies.predictives.mobo.ExplicitReferencePoint.values",
    "bofire.data_models.strategies.predictives.mobo.FixedReferenceValue.value",
    "bofire.data_models.strategies.predictives.mobo.MoboStrategy.acquisition_function",
    "bofire.data_models.strategies.predictives.mobo.MoboStrategy.ref_point",
    "bofire.data_models.strategies.predictives.mobo.RelativeMovingReferenceValue.orient_at_best",
    "bofire.data_models.strategies.predictives.mobo.RelativeMovingReferenceValue.scaling",
    "bofire.data_models.strategies.predictives.mobo.RelativeToMaxMovingReferenceValue.orient_at_best",
    "bofire.data_models.strategies.predictives.mobo.RelativeToMaxMovingReferenceValue.scaling",
    "bofire.data_models.strategies.predictives.multi_fidelity.MultiFidelityVarianceBasedStrategy.fidelity_thresholds",
    "bofire.data_models.strategies.predictives.multi_fidelity_knowledge_gradient.MultiFidelityHVKGStrategy.acquisition_function",
    "bofire.data_models.strategies.predictives.multi_fidelity_knowledge_gradient.MultiFidelityHVKGStrategy.fidelity_cost_model_spec",
    "bofire.data_models.strategies.predictives.predictive.PredictiveStrategy.convergence_criterion",
    "bofire.data_models.strategies.predictives.qparego.QparegoStrategy.acquisition_function",
    "bofire.data_models.strategies.predictives.sobo.AdditiveSoboStrategy.use_output_constraints",
    "bofire.data_models.strategies.predictives.sobo.CustomSoboStrategy.dump",
    "bofire.data_models.strategies.predictives.sobo.CustomSoboStrategy.use_output_constraints",
    "bofire.data_models.strategies.predictives.sobo.MultiplicativeAdditiveSoboStrategy.additive_features",
    "bofire.data_models.strategies.predictives.sobo.MultiplicativeAdditiveSoboStrategy.use_output_constraints",
    "bofire.data_models.strategies.predictives.sobo.SoboBaseStrategy.acquisition_function",
    "bofire.data_models.strategies.random.RandomStrategy.fallback_sampling_method",
    "bofire.data_models.strategies.random.RandomStrategy.max_combinations",
    "bofire.data_models.strategies.random.RandomStrategy.max_iters",
    "bofire.data_models.strategies.random.RandomStrategy.n_burnin",
    "bofire.data_models.strategies.random.RandomStrategy.n_thinning",
    "bofire.data_models.strategies.random.RandomStrategy.nchoosek_max_iters",
    "bofire.data_models.strategies.random.RandomStrategy.num_base_samples",
    "bofire.data_models.strategies.random.RandomStrategy.sampler_kwargs",
    "bofire.data_models.strategies.shortest_path.ShortestPathStrategy.atol",
    "bofire.data_models.strategies.shortest_path.ShortestPathStrategy.end",
    "bofire.data_models.strategies.shortest_path.ShortestPathStrategy.start",
    "bofire.data_models.strategies.stepwise.conditions.CombiCondition.conditions",
    "bofire.data_models.strategies.stepwise.conditions.CombiCondition.n_required_conditions",
    "bofire.data_models.strategies.stepwise.conditions.FeasibleExperimentCondition.n_required_feasible_experiments",
    "bofire.data_models.strategies.stepwise.conditions.FeasibleExperimentCondition.threshold",
    "bofire.data_models.strategies.stepwise.conditions.NumberOfExperimentsCondition.n_experiments",
    "bofire.data_models.strategies.stepwise.stepwise.Step.condition",
    "bofire.data_models.strategies.stepwise.stepwise.Step.strategy_data",
    "bofire.data_models.strategies.stepwise.stepwise.Step.transform",
    "bofire.data_models.strategies.stepwise.stepwise.StepwiseStrategy.steps",
    "bofire.data_models.strategies.strategy.Strategy.domain",
    "bofire.data_models.strategies.strategy.Strategy.seed",
    "bofire.data_models.surrogates.deterministic.CategoricalDeterministicSurrogate.mapping",
    "bofire.data_models.surrogates.deterministic.LinearDeterministicSurrogate.coefficients",
    "bofire.data_models.surrogates.deterministic.LinearDeterministicSurrogate.intercept",
    "bofire.data_models.surrogates.pairwise_gp.PairwiseGPSurrogate.kernel",
    "bofire.data_models.surrogates.pairwise_gp.PairwiseGPSurrogate.likelihood",
    "bofire.data_models.surrogates.pairwise_gp.PairwiseGPSurrogate.scaler",
}

# Classes that do not yet carry a docstring of their own. Only ever remove entries.
UNDOCUMENTED_CLASSES = {
    "bofire.data_models.strategies.doe.AOptimalityCriterion",
    "bofire.data_models.strategies.doe.DOptimalityCriterion",
    "bofire.data_models.strategies.doe.DoEOptimalityCriterion",
    "bofire.data_models.strategies.doe.DoEStrategy",
    "bofire.data_models.strategies.doe.EOptimalityCriterion",
    "bofire.data_models.strategies.doe.GOptimalityCriterion",
    "bofire.data_models.strategies.doe.IOptimalityCriterion",
    "bofire.data_models.strategies.doe.KOptimalityCriterion",
    "bofire.data_models.strategies.doe.OptimalityCriterion",
    "bofire.data_models.strategies.doe.SpaceFillingCriterion",
    "bofire.data_models.strategies.predictives.acqf_optimization.AcquisitionOptimizer",
    "bofire.data_models.strategies.predictives.acqf_optimization.BotorchOptimizer",
    "bofire.data_models.strategies.predictives.botorch.BotorchStrategy",
    "bofire.data_models.strategies.predictives.enting.EntingStrategy",
    "bofire.data_models.strategies.predictives.mobo.MoboStrategy",
    "bofire.data_models.strategies.predictives.mobo.ReferencePoint",
    "bofire.data_models.strategies.predictives.mobo.ReferenceValue",
    "bofire.data_models.strategies.predictives.multi_fidelity.MultiFidelityVarianceBasedStrategy",
    "bofire.data_models.strategies.predictives.multi_fidelity_knowledge_gradient.MultiFidelityHVKGStrategy",
    "bofire.data_models.strategies.predictives.multiobjective.MultiobjectiveStrategy",
    "bofire.data_models.strategies.predictives.predictive.PredictiveStrategy",
    "bofire.data_models.strategies.predictives.qparego.QparegoStrategy",
    "bofire.data_models.strategies.predictives.sobo.AdditiveSoboStrategy",
    "bofire.data_models.strategies.predictives.sobo.CustomSoboStrategy",
    "bofire.data_models.strategies.predictives.sobo.MultiplicativeSoboStrategy",
    "bofire.data_models.strategies.predictives.sobo.SoboBaseStrategy",
    "bofire.data_models.strategies.predictives.sobo.SoboStrategy",
    "bofire.data_models.strategies.stepwise.conditions.AlwaysTrueCondition",
    "bofire.data_models.strategies.stepwise.conditions.CombiCondition",
    "bofire.data_models.strategies.stepwise.conditions.Condition",
    "bofire.data_models.strategies.stepwise.conditions.NumberOfExperimentsCondition",
    "bofire.data_models.strategies.stepwise.conditions.SingleCondition",
    "bofire.data_models.strategies.stepwise.stepwise.Step",
    "bofire.data_models.strategies.stepwise.stepwise.StepwiseStrategy",
    "bofire.data_models.strategies.strategy.Strategy",
    "bofire.data_models.surrogates.deterministic.LinearDeterministicSurrogate",
    "bofire.data_models.surrogates.empirical.EmpiricalSurrogate",
}


def _collect_data_model_classes() -> Dict[str, Type[BaseModel]]:
    """Return every pydantic model declared under `bofire.data_models`, keyed by path.

    Import errors are deliberately not caught: a module that cannot be imported would
    otherwise drop out of the parametrization and leave a whole package unenforced.
    """
    classes: Dict[str, Type[BaseModel]] = {}
    for module_info in pkgutil.walk_packages(
        bofire.data_models.__path__,
        bofire.data_models.__name__ + ".",
    ):
        module = importlib.import_module(module_info.name)
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj.__module__.startswith("bofire.data_models")
                # parametrized generics such as `DataFrame[CandidateRow]` are created by
                # pydantic, not declared; their generic carries the docs
                and obj.__pydantic_generic_metadata__["origin"] is None
            ):
                classes[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return classes


DATA_MODEL_CLASSES = _collect_data_model_classes()


def _own_field_names(cls: Type[BaseModel]) -> List[str]:
    """Return the fields declared on `cls` itself, ignoring the `type` discriminator.

    Inherited fields are documented on the class that declares them. Checking them
    again here would report one undocumented base class field once per subclass.
    """
    annotations = getattr(cls, "__annotations__", {})
    return [name for name in cls.model_fields if name in annotations and name != "type"]


def _undocumented_fields() -> set:
    return {
        f"{path}.{name}"
        for path, cls in DATA_MODEL_CLASSES.items()
        for name in _own_field_names(cls)
        if not cls.model_fields[name].description
    }


def _undocumented_classes() -> set:
    return {
        path
        for path, cls in DATA_MODEL_CLASSES.items()
        if not cls.__dict__.get("__doc__")
    }


@pytest.mark.parametrize("path", sorted(DATA_MODEL_CLASSES))
def test_own_fields_have_descriptions(path: str):
    cls = DATA_MODEL_CLASSES[path]
    missing = {
        f"{path}.{name}"
        for name in _own_field_names(cls)
        if not cls.model_fields[name].description
    }
    assert not missing - UNDOCUMENTED_FIELDS, (
        f"{sorted(missing - UNDOCUMENTED_FIELDS)} lack `Field(description=...)`. "
        "Field prose belongs in the field, not in an `Attributes:` docstring block, "
        "because only the former reaches `model_json_schema()`."
    )


@pytest.mark.parametrize("path", sorted(DATA_MODEL_CLASSES))
def test_classes_have_docstrings(path: str):
    cls = DATA_MODEL_CLASSES[path]
    if cls.__dict__.get("__doc__"):
        return
    assert path in UNDOCUMENTED_CLASSES, (
        f"{path} has no docstring of its own. State what it is and when to pick it "
        "over its siblings in the union."
    )


def test_field_allowlist_has_no_stale_entries():
    stale = UNDOCUMENTED_FIELDS - _undocumented_fields()
    assert not stale, (
        f"{sorted(stale)} are documented (or no longer exist) but still listed in "
        "UNDOCUMENTED_FIELDS. Delete them: the allowlist may only shrink."
    )


def test_class_allowlist_has_no_stale_entries():
    stale = UNDOCUMENTED_CLASSES - _undocumented_classes()
    assert not stale, (
        f"{sorted(stale)} are documented (or no longer exist) but still listed in "
        "UNDOCUMENTED_CLASSES. Delete them: the allowlist may only shrink."
    )
