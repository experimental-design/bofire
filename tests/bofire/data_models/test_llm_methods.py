"""Tests for to_pydantic_field(), to_description(), and to_pydantic_model() methods."""

from typing import Literal

import pytest
from pydantic import ValidationError

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductEqualityConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.features.molecular import ContinuousMolecularInput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective


# --- to_pydantic_field tests ---


class TestContinuousInputToPydanticField:
    def test_basic(self):
        feat = ContinuousInput(key="temp", bounds=(20.0, 200.0))
        field_type, field_info = feat.to_pydantic_field()
        assert field_type is float
        assert field_info.metadata[0].ge == 20.0
        assert field_info.metadata[1].le == 200.0
        assert "bounds [20.0, 200.0]" in field_info.description

    def test_with_context(self):
        feat = ContinuousInput(
            key="temp", bounds=(20.0, 200.0), context="Temperature in C"
        )
        _, field_info = feat.to_pydantic_field()
        assert "Temperature in C" in field_info.description

    def test_allow_zero(self):
        feat = ContinuousInput(key="x", bounds=(0.01, 0.5), allow_zero=True)
        _, field_info = feat.to_pydantic_field()
        assert field_info.metadata[0].ge == 0.0
        assert "can also be 0" in field_info.description


class TestDiscreteInputToPydanticField:
    def test_basic(self):
        feat = DiscreteInput(key="n", values=[1.0, 2.0, 5.0])
        field_type, field_info = feat.to_pydantic_field()
        assert field_type == Literal[1.0, 2.0, 5.0]
        assert "allowed values" in field_info.description


class TestCategoricalInputToPydanticField:
    def test_basic(self):
        feat = CategoricalInput(key="sol", categories=["water", "ethanol", "toluene"])
        field_type, _ = feat.to_pydantic_field()
        assert field_type == Literal["water", "ethanol", "toluene"]

    def test_respects_allowed(self):
        feat = CategoricalInput(
            key="sol",
            categories=["water", "ethanol", "toluene"],
            allowed=[True, True, False],
        )
        field_type, field_info = feat.to_pydantic_field()
        assert field_type == Literal["water", "ethanol"]
        assert "toluene" not in field_info.description


class TestCategoricalMolecularInputToPydanticField:
    def test_smiles_in_description(self):
        feat = CategoricalMolecularInput(key="mol", categories=["CCO", "CC"])
        _, field_info = feat.to_pydantic_field()
        assert "SMILES" in field_info.description
        assert "CCO" in field_info.description


class TestContinuousMolecularInputToPydanticField:
    def test_smiles_in_description(self):
        feat = ContinuousMolecularInput(key="conc", molecule="CCO", bounds=(0.0, 1.0))
        _, field_info = feat.to_pydantic_field()
        assert "SMILES: CCO" in field_info.description


class TestCategoricalDescriptorInputToPydanticField:
    def test_descriptors_in_description(self):
        feat = CategoricalDescriptorInput(
            key="cat",
            categories=["a", "b"],
            descriptors=["d1", "d2"],
            values=[[1.0, 2.0], [3.0, 4.0]],
        )
        _, field_info = feat.to_pydantic_field()
        assert "descriptors per category" in field_info.description
        assert "d1" in field_info.description


# --- Inputs.to_pydantic_model tests ---


class TestInputsToPydanticModel:
    def test_creates_valid_model(self):
        inputs = Inputs(
            features=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                CategoricalInput(key="x2", categories=["a", "b"]),
            ]
        )
        Model = inputs.to_pydantic_model()
        schema = Model.model_json_schema()
        assert "x1" in schema["properties"]
        assert "x2" in schema["properties"]
        assert schema["properties"]["x1"]["type"] == "number"

    def test_model_validates_bounds(self):
        inputs = Inputs(features=[ContinuousInput(key="x", bounds=(0, 1))])
        Model = inputs.to_pydantic_model()
        obj = Model(x=0.5)
        assert obj.x == 0.5
        with pytest.raises(ValidationError):
            Model(x=5.0)

    def test_model_validates_categories(self):
        inputs = Inputs(
            features=[
                CategoricalInput(key="c", categories=["a", "b"], allowed=[True, False])
            ]
        )
        Model = inputs.to_pydantic_model()
        obj = Model(c="a")
        assert obj.c == "a"
        with pytest.raises(ValidationError):
            Model(c="b")


# --- Constraint.to_description tests ---


class TestConstraintToDescription:
    def test_linear_equality(self):
        c = LinearEqualityConstraint(
            features=["x1", "x2"], coefficients=[1.0, 2.0], rhs=5.0
        )
        desc = c.to_description()
        assert "1.0*x1" in desc
        assert "2.0*x2" in desc
        assert "= 5.0" in desc

    def test_linear_inequality(self):
        c = LinearInequalityConstraint(
            features=["x1", "x2"], coefficients=[1.0, 2.0], rhs=5.0
        )
        desc = c.to_description()
        assert "<= 5.0" in desc

    def test_linear_with_context(self):
        c = LinearInequalityConstraint(
            features=["x1", "x2"],
            coefficients=[1.0, 2.0],
            rhs=5.0,
            context="Safety limit",
        )
        desc = c.to_description()
        assert "Safety limit" in desc

    def test_nchoosek(self):
        c = NChooseKConstraint(
            features=["x1", "x2", "x3"],
            min_count=1,
            max_count=2,
            none_also_valid=False,
        )
        desc = c.to_description()
        assert "Choose 1-2" in desc
        assert "x1" in desc

    def test_nchoosek_none_valid(self):
        c = NChooseKConstraint(
            features=["x1", "x2"],
            min_count=1,
            max_count=2,
            none_also_valid=True,
        )
        desc = c.to_description()
        assert "or none" in desc

    def test_product_equality(self):
        c = ProductEqualityConstraint(
            features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
        )
        desc = c.to_description()
        assert "x1^2" in desc
        assert "= 1.0" in desc

    def test_product_inequality(self):
        c = ProductInequalityConstraint(
            features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
        )
        desc = c.to_description()
        assert "<= 1.0" in desc


# --- Objective.to_description tests ---


class TestObjectiveToDescription:
    def test_maximize(self):
        assert MaximizeObjective(w=1.0).to_description() == "Maximize"

    def test_minimize(self):
        assert MinimizeObjective(w=1.0).to_description() == "Minimize"


# --- Output.to_description tests ---


class TestOutputToDescription:
    def test_continuous_output(self):
        feat = ContinuousOutput(
            key="yield",
            objective=MaximizeObjective(w=1.0),
            context="Target >90%",
        )
        desc = feat.to_description()
        assert "yield" in desc
        assert "Maximize" in desc
        assert "Target >90%" in desc

    def test_continuous_output_no_context(self):
        feat = ContinuousOutput(key="yield", objective=MinimizeObjective(w=1.0))
        desc = feat.to_description()
        assert "yield" in desc
        assert "Minimize" in desc


# --- Domain.to_description tests ---


class TestDomainToDescription:
    def test_basic(self):
        domain = Domain.from_lists(
            inputs=[ContinuousInput(key="x", bounds=(0, 1))],
            outputs=[ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))],
        )
        desc = domain.to_description()
        assert "Objectives" in desc
        assert "y" in desc
        assert "Maximize" in desc

    def test_with_context(self):
        domain = Domain.from_lists(
            inputs=[ContinuousInput(key="x", bounds=(0, 1))],
            outputs=[ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))],
        )
        domain.context = "Optimizing a reaction"
        desc = domain.to_description()
        assert "Optimizing a reaction" in desc

    def test_with_constraints(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
            ],
            outputs=[ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))],
            constraints=[
                LinearInequalityConstraint(
                    features=["x1", "x2"],
                    coefficients=[1.0, 1.0],
                    rhs=1.5,
                    context="Budget constraint",
                )
            ],
        )
        desc = domain.to_description()
        assert "Constraints" in desc
        assert "<= 1.5" in desc
        assert "Budget constraint" in desc


# --- Context field tests ---


class TestContextFields:
    def test_feature_context(self):
        feat = ContinuousInput(key="x", bounds=(0, 1), context="Temperature")
        assert feat.context == "Temperature"
        dumped = feat.model_dump()
        assert dumped["context"] == "Temperature"

    def test_feature_context_default_none(self):
        feat = ContinuousInput(key="x", bounds=(0, 1))
        assert feat.context is None

    def test_constraint_context(self):
        c = LinearInequalityConstraint(
            features=["x1", "x2"],
            coefficients=[1.0, 1.0],
            rhs=1.0,
            context="Safety",
        )
        assert c.context == "Safety"

    def test_domain_context(self):
        domain = Domain.from_lists(
            inputs=[ContinuousInput(key="x", bounds=(0, 1))],
            outputs=[ContinuousOutput(key="y")],
        )
        domain.context = "Test problem"
        assert domain.context == "Test problem"
        dumped = domain.model_dump()
        assert dumped["context"] == "Test problem"

    def test_context_roundtrip(self):
        domain = Domain.from_lists(
            inputs=[ContinuousInput(key="x", bounds=(0, 1), context="Feature ctx")],
            outputs=[ContinuousOutput(key="y", context="Output ctx")],
        )
        domain.context = "Domain ctx"
        dumped = domain.model_dump()
        restored = Domain(**dumped)
        assert restored.context == "Domain ctx"
        assert restored.inputs[0].context == "Feature ctx"
        assert restored.outputs[0].context == "Output ctx"
