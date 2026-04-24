"""Centered gradient program sampler strategy.

Samples monotone HPLC gradient programs using the Option-B endpoint approach,
correcting the diagonal bias that BoFire's hit-and-run produces on polytopes
with ordering constraints.

Design
------
All phi/t bounds are read directly from the domain's ContinuousInput feature bounds,
which must be created via ``create_absolute_domain(OptimizationSpec)``:

    phi_0   : bounds = (phi_min, phi_0_max)
    phi_1..{n-1}: bounds = (phi_min, phi_max)
    phi_{n} : bounds = (phi_last_min, phi_max)
    t_1..t_{n-1}: bounds = (t_min, t_max - eps)
    t_{n}   : bounds = (t_last_min, t_max)   [present only in variable-duration mode]

When ``t_{n}`` is absent from the domain (fixed-duration mode), the fixed end
time must be supplied via the ``t_n_fixed`` field of the data model.
"""

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.features.api import ContinuousInput
from bofire.strategies.strategy import Strategy


class UniformGradientStrategy(Strategy):
    """Strategy that samples gradient programs with ``sample_gradient_programs_uniform``.

    Parameters are inferred from the domain feature bounds — no redundant fields.
    """

    def __init__(self, data_model: data_models.UniformGradientStrategy, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        # Parse sampling parameters from the domain once at construction time
        self._params = self._parse_domain()

    # ── domain introspection ───────────────────────────────────────────────

    def _parse_domain(self) -> dict:
        """Extract uniform-sampler parameters from the domain feature bounds.

        Returns a dict with keys:
            n_nodes, phi_min, phi_0_max, phi_last_min, phi_max,
            t_min, t_max, t_last_min (or None for fixed-duration),
            t_interior_upper, eps
        """
        from bofire.data_models.constraints.api import LinearInequalityConstraint

        inputs = self.domain.inputs
        keys = inputs.get_keys(ContinuousInput)

        # Determine n_nodes from phi column count
        phi_indices = (int(k.split("_")[1]) for k in keys if k.startswith("phi_"))
        n_nodes = max(phi_indices)  # phi_0 .. phi_{n_nodes}

        t_interior_keys = [f"t_{i}" for i in range(1, n_nodes)]
        t_interior_upper = min(
            self.domain.inputs.get_by_key(k).upper_bound for k in t_interior_keys
        )

        # Detect eps from t-ordering constraints (rhs = -eps).
        # Fall back to 0.01 if no such constraint exists.
        eps = 0.01
        for c in self.domain.constraints:
            if (
                isinstance(c, LinearInequalityConstraint)
                and len(c.features) == 2
                and c.features[0].startswith("t_")
                and c.features[1].startswith("t_")
                and c.rhs < 0
            ):
                eps = -c.rhs
                break

        return {
            "n_nodes": n_nodes,
            "phi_min": self.domain.inputs.get_by_key("phi_0").lower_bound,
            "phi_0_max": self.domain.inputs.get_by_key("phi_0").upper_bound,
            "phi_last_min": self.domain.inputs.get_by_key(f"phi_{n_nodes}").lower_bound,
            "phi_max": self.domain.inputs.get_by_key(f"phi_{n_nodes}").upper_bound,
            "t_min": self.domain.inputs.get_by_key("t_1").lower_bound,
            "t_max": self.domain.inputs.get_by_key(f"t_{n_nodes}").upper_bound,
            "t_last_min": self.domain.inputs.get_by_key(f"t_{n_nodes}").lower_bound,
            "t_interior_upper": t_interior_upper,
            "eps": eps,
        }

    # ── sampling (core logic) ──────────────────────────────────────────────

    @staticmethod
    def _sample(
        n_programs: int,
        n_nodes: int,
        phi_min: float,
        phi_max: float,
        phi_0_max: float,
        phi_last_min: float,
        t_min: float,
        t_max: float,
        t_last_min: Optional[float],
        t_interior_upper: float,
        eps: float = 0.01,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Option-B centered monotone gradient program sampler.

        phi sampling
        ------------
        phi_0 ~ U[phi_min, phi_0_max]
        phi_n ~ U[phi_last_min, phi_max]
        Constraint phi_0 < phi_n enforced by rejection sampling with a
        conditional-resample fallback (never swap — that breaks endpoint bounds).
        Interior knots: sorted uniforms in [phi_0, phi_n].

        t sampling
        ----------
        t_last_min is None  → fixed-duration: T = t_max for every program.
        t_last_min < t_max  → variable-duration: T ~ U[t_last_min, t_max].
        Interior knots t_1..t_{n-1}: sorted uniforms in [t_min, T).
        Final knot t_n pinned to T.
        """
        rng = np.random.default_rng(seed)

        # ── phi endpoints ──────────────────────────────────────────────────
        phi_0 = rng.uniform(phi_min, phi_0_max, size=n_programs)
        phi_n = rng.uniform(phi_last_min, phi_max, size=n_programs)

        bad = phi_0 >= phi_n
        n_bad = int(bad.sum())
        iters = 0
        while n_bad > 0:
            if iters >= 100:
                # Fallback: conditionally resample phi_0 so that
                # phi_0 < phi_n  AND  phi_0 <= phi_0_max are both satisfied.
                _eps = 1e-6
                hi = np.minimum(phi_0_max, phi_n[bad] - _eps)
                hi = np.maximum(hi, phi_min + _eps)
                phi_0[bad] = phi_min + rng.uniform(0, 1, size=n_bad) * (hi - phi_min)
                break
            phi_0[bad] = rng.uniform(phi_min, phi_0_max, size=n_bad)
            phi_n[bad] = rng.uniform(phi_last_min, phi_max, size=n_bad)
            bad = phi_0 >= phi_n
            n_bad = int(bad.sum())
            iters += 1

        # ── phi interior knots ────────────────────────────────────────────
        n_interior = n_nodes - 1
        if n_interior > 0:
            u_sorted = np.sort(rng.uniform(0, 1, size=(n_programs, n_interior)), axis=1)
            phi_interior = phi_0[:, None] + (phi_n - phi_0)[:, None] * u_sorted
            phi_all = np.concatenate(
                [phi_0[:, None], phi_interior, phi_n[:, None]], axis=1
            )
        else:
            phi_all = np.concatenate([phi_0[:, None], phi_n[:, None]], axis=1)

        # ── t: sample total duration, then interior knots ─────────────────
        T = rng.uniform(t_last_min, t_max, size=n_programs)

        if n_nodes > 1:
            n_interior = n_nodes - 1
            # Guarantee minimum gap of `eps` between every adjacent pair of knots
            # (including the gap from t_{n-1} to T=t_n).
            # Strategy: shrink the available range by n_interior * eps, sample
            # sorted uniforms in that reduced space, then add a deterministic
            # offset k*eps to the k-th knot.  This gives:
            #   t_{k+1} - t_k  >=  eps  for all k
            #   t_{n-1}        <=  T - eps
            # The marginal distribution of each knot is still approximately
            # uniform over the valid subspace.
            # Also cap the effective endpoint used for interior sampling so that
            # every interior t_i stays within its own feature upper bound.
            # This is essential when fixed-duration mode pins T beyond those
            # interior bounds (e.g. t_n fixed at a larger value).
            T_for_interior = np.minimum(T, t_interior_upper + eps)
            T_adj = (
                T_for_interior - n_interior * eps
            )  # adjusted upper bound for raw samples
            # Guard: if the range is too narrow, collapse all interior knots to t_min
            valid = T_adj > t_min
            T_adj = np.where(valid, T_adj, t_min)

            v_sorted = np.sort(rng.uniform(0, 1, size=(n_programs, n_interior)), axis=1)
            t_interior_raw = t_min + v_sorted * (T_adj[:, None] - t_min)
            # add offset: 0, eps, 2*eps, ..., (n_interior-1)*eps
            t_interior = t_interior_raw + eps * np.arange(n_interior)
            t_all = np.concatenate([t_interior, T[:, None]], axis=1)
        else:
            t_all = T[:, None]

        phi_cols = {f"phi_{i}": phi_all[:, i] for i in range(n_nodes + 1)}
        t_cols = {f"t_{i+1}": t_all[:, i] for i in range(n_nodes)}
        return pd.DataFrame({**phi_cols, **t_cols})

    # ── Strategy interface ─────────────────────────────────────────────────

    def has_sufficient_experiments(self) -> bool:
        return True

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        p = self._params
        return self._sample(
            n_programs=candidate_count,
            n_nodes=p["n_nodes"],
            phi_min=p["phi_min"],
            phi_max=p["phi_max"],
            phi_0_max=p["phi_0_max"],
            phi_last_min=p["phi_last_min"],
            t_min=p["t_min"],
            t_max=p["t_max"],
            t_last_min=p["t_last_min"],
            t_interior_upper=p["t_interior_upper"],
            eps=p["eps"],
            seed=self._get_seed(),
        )
