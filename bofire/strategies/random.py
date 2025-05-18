import math
import warnings
from copy import deepcopy
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
import torch
from botorch.optim.initializers import sample_q_batches_from_polytope
from botorch.optim.parameter_constraints import _generate_unfixed_lin_constraints
from pydantic.types import PositiveInt
from typing_extensions import Self

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    AnyContinuousConstraint,
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.strategies.strategy import Strategy, make_strategy
from bofire.utils.torch_tools import (
    get_interpoint_constraints,
    get_linear_constraints,
    tkwargs,
)


class RandomStrategy(Strategy):
    """Strategy for randomly selecting new candidates.

    This strategy generates candidate samples using the random strategy. It first checks if the domain
    is compatible with polytope sampling. If so, it uses polytope sampling to generate candidate
    samples. If not, it performs rejection sampling by repeatedly generating candidates with polytope sampling
    until the desired number of valid samples is obtained.

    Args:
        data_model (data_models.RandomStrategy): The data model for the random strategy.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        data_model: data_models.RandomStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.num_base_samples = data_model.num_base_samples
        self.max_iters = data_model.max_iters
        self.fallback_sampling_method = data_model.fallback_sampling_method
        self.n_burnin = data_model.n_burnin
        self.n_thinning = data_model.n_thinning

    def has_sufficient_experiments(self) -> bool:
        """Check if there are sufficient experiments for the strategy.

        Returns:
            bool: True if there are sufficient experiments, False otherwise.

        """
        return True

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:  # type: ignore
        """Generate candidate samples using the random strategy.

        If the domain is compatible with polytope sampling, it uses the polytope sampling to generate
        candidate samples. Otherwise, it performs rejection sampling by repeatedly generating candidate
        samples until the desired number of valid samples is obtained.

        Args:
            candidate_count (PositiveInt): The number of candidate samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the generated candidate samples.

        """
        # no nonlinear constraints present --> no rejection sampling needed
        if len(self.domain.constraints) == len(
            self.domain.constraints.get(
                [
                    LinearInequalityConstraint,
                    LinearEqualityConstraint,
                    NChooseKConstraint,
                    InterpointEqualityConstraint,
                ],
            ),
        ):
            return self._sample_with_nchooseks(candidate_count)
        # perform the rejection sampling
        num_base_samples = self.num_base_samples or candidate_count
        n_iters = 0
        n_found = 0
        valid_samples = []
        while n_found < candidate_count:
            if n_iters > self.max_iters:
                raise ValueError("Maximum iterations exceeded in rejection sampling.")
            samples = self._sample_with_nchooseks(num_base_samples)
            valid = self.domain.constraints.is_fulfilled(samples)
            n_found += np.sum(valid)
            valid_samples.append(samples[valid])
            n_iters += 1
        return pd.concat(valid_samples, ignore_index=True).iloc[:candidate_count]

    def _sample_with_nchooseks(
        self,
        candidate_count: int,
    ) -> pd.DataFrame:
        """Sample from the domain with NChooseK constraints.

        Args:
            candidate_count (int): The number of samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled data.

        """
        if len(self.domain.constraints.get(NChooseKConstraint)) > 0:
            _, unused = self.domain.get_nchoosek_combinations()

            if candidate_count <= len(unused):
                sampled_combinations = [
                    unused[i]
                    for i in np.random.default_rng(self._get_seed()).choice(
                        len(unused),
                        size=candidate_count,
                        replace=False,
                    )
                ]
                num_samples_per_it = 1
            else:
                sampled_combinations = unused
                num_samples_per_it = math.ceil(candidate_count / len(unused))

            samples = []
            for u in sampled_combinations:
                # create new domain without the nchoosekconstraints
                domain = deepcopy(self.domain)
                domain.constraints = domain.constraints.get(excludes=NChooseKConstraint)
                # fix the unused features
                for key in u:
                    feat = domain.inputs.get_by_key(key=key)
                    assert isinstance(feat, ContinuousInput)
                    feat.bounds = [0.0, 0.0]
                # setup then sampler for this situation
                samples.append(
                    self._sample_from_polytope(
                        domain=domain,
                        fallback_sampling_method=self.fallback_sampling_method,
                        n_burnin=self.n_burnin,
                        n_thinning=self.n_thinning,
                        seed=self._get_seed(),
                        n=num_samples_per_it,
                    ),
                )
            samples = pd.concat(samples, axis=0, ignore_index=True)
            return samples.sample(
                n=candidate_count,
                replace=False,
                ignore_index=True,
                random_state=self._get_seed(),
            )

        return self._sample_from_polytope(
            domain=self.domain,
            fallback_sampling_method=self.fallback_sampling_method,
            n_burnin=self.n_burnin,
            n_thinning=self.n_thinning,
            seed=self._get_seed(),
            n=candidate_count,
        )

    @staticmethod
    def _sample_from_polytope(
        domain: Domain,
        n: int,
        fallback_sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM,
        n_burnin: int = 1000,
        n_thinning: int = 32,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Sample points from a polytope defined by the given domain.

        Args:
            n (int): The number of points to sample.
            domain (Domain): The domain defining the polytope.
            fallback_sampling_method (SamplingMethodEnum, optional): The fallback sampling method to use when the domain has no constraints.
                Defaults to SamplingMethodEnum.UNIFORM.
            n_burnin (int, optional): The number of burn-in samples for the polytope sampler. Defaults to 1000.
            n_thinning (int, optional): The thinning factor for the polytope sampler. Defaults to 32.
            seed (Optional[int], optional): The seed value for random number generation. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled points.

        """
        if seed is None:
            seed = np.random.default_rng().integers(1, 1000000)

        # here we have to adapt for categoricals
        if len(domain.constraints.get(AnyContinuousConstraint)) == 0:  # type: ignore
            return domain.inputs.sample(n, fallback_sampling_method, seed=seed)

        # check if we have pseudo fixed features in the linear equality constraints
        # a pseudo fixed is a linear euquality constraint with only one feature included
        # this can happen when fixing features when sampling with NChooseK constraints
        eqs = get_linear_constraints(
            domain=domain,
            constraint=LinearEqualityConstraint,
            unit_scaled=False,
        )
        cleaned_eqs = []
        fixed_features: Dict[str, float] = {  # type: ignore
            feat.key: feat.fixed_value()[0]  # type: ignore
            for feat in domain.inputs.get(ContinuousInput)
            if feat.is_fixed()
        }

        for eq in eqs:
            if (
                len(eq[0]) == 1
            ):  # only one coefficient, so this is a pseudo fixed feature
                fixed_features[domain.inputs.get_keys(ContinuousInput)[eq[0][0]]] = (
                    float(eq[2] / eq[1][0])
                )
            else:
                cleaned_eqs.append(eq)

        fixed_features_indices: Dict[int, float] = {
            domain.inputs.get_keys(ContinuousInput).index(key): value
            for key, value in fixed_features.items()
        }

        ineqs = get_linear_constraints(
            domain=domain,
            constraint=LinearInequalityConstraint,
            unit_scaled=False,
        )

        interpoints = get_interpoint_constraints(domain=domain, n_candidates=n)

        lower = [
            feat.lower_bound  # type: ignore
            for feat in domain.inputs.get(ContinuousInput)
            if feat.key not in fixed_features
        ]

        upper = [
            feat.upper_bound  # type: ignore
            for feat in domain.inputs.get(ContinuousInput)
            if feat.key not in fixed_features
        ]

        if len(lower) == 0:
            warnings.warn(
                "Nothing to sample, all is fixed. Just the fixed set is returned.",
                UserWarning,
            )
            samples = pd.DataFrame(
                data=np.nan,
                index=range(n),
                columns=domain.inputs.get_keys(),
            )
        else:
            bounds = torch.tensor([lower, upper]).to(**tkwargs)

            unfixed_ineqs = _generate_unfixed_lin_constraints(
                constraints=ineqs,
                eq=False,
                fixed_features=fixed_features_indices,
                dimension=len(domain.inputs.get(ContinuousInput)),
            )
            unfixed_eqs = _generate_unfixed_lin_constraints(
                constraints=cleaned_eqs,
                eq=True,
                fixed_features=fixed_features_indices,
                dimension=len(domain.inputs.get(ContinuousInput)),
            )
            unfixed_interpoints = _generate_unfixed_lin_constraints(
                constraints=interpoints,
                eq=True,
                fixed_features=fixed_features_indices,
                dimension=len(domain.inputs.get(ContinuousInput)),
            )

            combined_eqs = unfixed_eqs + unfixed_interpoints  # type: ignore

            # now use the hit and run sampler
            candidates = sample_q_batches_from_polytope(
                n=1,
                q=n,
                bounds=bounds.to(**tkwargs),
                inequality_constraints=(
                    unfixed_ineqs if len(unfixed_ineqs) > 0 else None  # type: ignore
                ),
                equality_constraints=combined_eqs if len(combined_eqs) > 0 else None,
                n_burnin=n_burnin,
                n_thinning=n_thinning,
                seed=seed,
            ).squeeze(dim=0)

            # check that the random generated candidates are not always the same
            if (candidates.unique(dim=0).shape[0] != n) and (n > 1):
                warnings.warn("Generated candidates are not unique!")

            free_continuals = [
                feat.key
                for feat in domain.inputs.get(ContinuousInput)
                if feat.key not in fixed_features
            ]
            # setup the output
            samples = pd.DataFrame(
                data=candidates.detach().numpy(),
                index=range(n),
                columns=free_continuals,
            )

        # setup the categoricals and discrete ones as uniform sampled vals
        samples = pd.concat(
            [
                samples,
                domain.inputs.get([CategoricalInput, DiscreteInput]).sample(
                    n,
                    method=fallback_sampling_method,
                    seed=seed,
                ),
            ],
            axis=1,
            ignore_index=False,
        )

        # setup the fixed continuous ones
        for key, value in fixed_features.items():
            samples[key] = value

        return samples[domain.inputs.get_keys()]

    @classmethod
    def make(
        cls,
        domain: Domain,
        fallback_sampling_method: SamplingMethodEnum | None = None,
        n_burnin: int | None = None,
        n_thinning: int | None = None,
        num_base_samples: int | None = None,
        max_iters: int | None = None,
        seed: int | None = None,
    ) -> Self:
        """Create a new instance of the RandomStrategy class.
        Args:
            domain: The domain we randomly sample from.
            fallback_sampling_method: The fallback sampling method to use when the domain has no constraints.
            n_burnin: The number of burn-in samples for the polytope sampler.
            n_thinning: The thinning factor for the polytope sampler.
            num_base_samples: The number of base samples for rejection sampling.
            max_iters: The maximum number of iterations for rejection sampling.
            seed: The seed value for random number generation.
        Returns:
            RandomStrategy: A new instance of the RandomStrategy class.
        """
        return cast(Self, make_strategy(cls, data_models.RandomStrategy, locals()))
