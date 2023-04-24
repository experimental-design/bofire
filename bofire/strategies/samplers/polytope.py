import warnings

import numpy as np
import pandas as pd
import torch
from botorch.utils.sampling import get_polytope_samples

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.strategies.api import PolytopeSampler as DataModel
from bofire.strategies.samplers.sampler import SamplerStrategy
from bofire.utils.torch_tools import get_linear_constraints, tkwargs


class PolytopeSampler(SamplerStrategy):
    """Sampler that generates samples from a Polytope defined by linear equality and ineqality constraints.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        fallback_sampling_method: SamplingMethodEnum, optional): Method to use for sampling when no
            constraints are present. Defaults to UNIFORM.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.fallback_sampling_method = data_model.fallback_sampling_method

    def _ask(self, n: int) -> pd.DataFrame:
        if len(self.domain.constraints) == 0:
            return self.domain.inputs.sample(n, self.fallback_sampling_method)

        # check if we have pseudo fixed features in the linear equality constraints
        # a pseude fixed is a linear euquality constraint with only one feature included
        # this can happen when fixing features when sampling with NChooseK constraints
        eqs = get_linear_constraints(
            domain=self.domain,
            constraint=LinearEqualityConstraint,  # type: ignore
            unit_scaled=False,
        )
        cleaned_eqs = []
        pseudo_fixed = {}
        for eq in eqs:
            if (
                len(eq[0]) == 1
            ):  # only one coefficient, so this is a pseudo fixed feature
                pseudo_fixed[
                    self.domain.inputs.get_keys(ContinuousInput)[eq[0][0]]
                ] = float(eq[2] / eq[1][0])
            else:
                cleaned_eqs.append(eq)

        # we have to map the indices in case of fixed features
        # as we remove all fixed feature for the sampler, we have to adjust the
        # indices in the constraints, here we get the mapper to map original
        # to adjusted indices
        feature_map = {}
        counter = 0
        for i, feat in enumerate(self.domain.get_features(ContinuousInput)):
            if (not feat.is_fixed()) and (feat.key not in pseudo_fixed.keys()):  # type: ignore
                feature_map[i] = counter
                counter += 1

        # get the bounds
        lower = [
            feat.lower_bound  # type: ignore
            for feat in self.domain.get_features(ContinuousInput)
            if not feat.is_fixed() and feat.key not in pseudo_fixed.keys()  # type: ignore
        ]
        if len(lower) == 0:
            warnings.warn(
                "Nothing to sample, all is fixed. Just the fixed set is returned.",
                UserWarning,
            )
            samples = pd.DataFrame(
                data=np.nan, index=range(n), columns=self.domain.inputs.get_keys()
            )
        else:
            upper = [
                feat.upper_bound  # type: ignore
                for feat in self.domain.get_features(ContinuousInput)
                if not feat.is_fixed() and feat.key not in pseudo_fixed.keys()  # type: ignore
            ]
            bounds = torch.tensor([lower, upper]).to(**tkwargs)
            assert bounds.shape[-1] == len(feature_map) == counter

            # get the inequality constraints and map features back
            # we also check that only features present in the mapper
            # are present in the constraints
            ineqs = get_linear_constraints(
                domain=self.domain,
                constraint=LinearInequalityConstraint,  # type: ignore
                unit_scaled=False,
            )
            for ineq in ineqs:
                for key, value in feature_map.items():
                    if key != value:
                        ineq[0][ineq[0] == key] = value
                assert (
                    ineq[0].max() <= counter
                ), "Something went wrong when transforming the linear constraints. Revisit the problem."

            # map the indice of the equality constraints
            for eq in cleaned_eqs:
                for key, value in feature_map.items():
                    if key != value:
                        eq[0][eq[0] == key] = value
                assert (
                    eq[0].max() <= counter
                ), "Something went wrong when transforming the linear constraints. Revisit the problem."

            # now use the hit and run sampler
            candidates = get_polytope_samples(
                n=n,
                bounds=bounds.to(**tkwargs),
                inequality_constraints=ineqs if len(ineqs) > 0 else None,
                equality_constraints=cleaned_eqs if len(cleaned_eqs) > 0 else None,
                n_burnin=1000,
                # thinning=200
            )

            # check that the random generated candidates are not always the same
            if (candidates.unique(dim=0).shape[0] != n) and (n > 1):
                raise ValueError("Generated candidates are not unique!")

            free_continuals = [
                feat.key
                for feat in self.domain.get_features(ContinuousInput)
                if not feat.is_fixed() and feat.key not in pseudo_fixed.keys()  # type: ignore
            ]

            # setup the output
            samples = pd.DataFrame(
                data=candidates.detach().numpy().reshape(n, len(free_continuals)),
                index=range(n),
                columns=free_continuals,
            )

        # setup the categoricals and discrete ones as uniform sampled vals
        for feat in self.domain.get_features([CategoricalInput, DiscreteInput]):
            samples[feat.key] = feat.sample(n)  # type: ignore

        # setup the fixed continuous ones
        for feat in self.domain.inputs.get_fixed():
            samples[feat.key] = feat.fixed_value()[0]  # type: ignore

        # setup the pseudo fixed ones
        for key, value in pseudo_fixed.items():
            samples[key] = value

        return samples

    def duplicate(self, domain: Domain) -> SamplerStrategy:
        data_model = DataModel(
            domain=domain,
            fallback_sampling_method=self.fallback_sampling_method,
        )
        return self.__class__(data_model=data_model)
