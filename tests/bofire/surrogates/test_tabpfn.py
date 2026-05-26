"""Smoke + autograd tests for the TabPFN surrogate.

Gated by ``pytest.importorskip("tabpfn")`` so CI without the extra still
collects. The tests look for a cached v2 checkpoint and skip if it isn't
present — TabPFN downloads gate behind a license-acceptance flow that
requires either ``TABPFN_TOKEN`` or an interactive browser.
"""

import os
from pathlib import Path

import pytest


pytest.importorskip("tabpfn")

import torch  # noqa: E402

import bofire.surrogates.api as surrogates  # noqa: E402
from bofire.benchmarks.single import Himmelblau  # noqa: E402
from bofire.data_models.surrogates.api import (
    TabPFNSurrogate as TabPFNDataModel,  # noqa: E402
)


def _cached_v2_checkpoint() -> str:
    candidates = [
        Path.home() / "Library/Caches/tabpfn/tabpfn-v2-regressor.ckpt",
        Path.home() / ".cache/tabpfn/tabpfn-v2-regressor.ckpt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    if os.environ.get("TABPFN_TOKEN"):
        return ""  # let the loader download
    pytest.skip(
        "No cached TabPFN v2 checkpoint found; set TABPFN_TOKEN to enable download."
    )


@pytest.fixture(scope="module")
def checkpoint() -> str:
    return _cached_v2_checkpoint()


def _toy_data(n: int = 20):
    bench = Himmelblau()
    experiments = bench.f(bench.domain.inputs.sample(n=n, seed=0), return_complete=True)
    return bench.domain, experiments


def _fit_surrogate(checkpoint: str, posterior_type: str = "gaussian"):
    domain, experiments = _toy_data()
    data_model = TabPFNDataModel(
        inputs=domain.inputs,
        outputs=domain.outputs,
        posterior_type=posterior_type,
        tabpfn_version="v2",
        checkpoint_path=checkpoint or None,
    )
    surrogate = surrogates.map(data_model)
    surrogate.fit(experiments)
    return surrogate, experiments


def test_fit_and_predict_gaussian(checkpoint):
    surrogate, experiments = _fit_surrogate(checkpoint, "gaussian")
    preds = surrogate.predict(experiments)
    assert preds.shape[0] == len(experiments)
    assert preds.notna().all().all()


def test_fit_and_predict_riemann(checkpoint):
    surrogate, experiments = _fit_surrogate(checkpoint, "riemann")
    preds = surrogate.predict(experiments)
    assert preds.shape[0] == len(experiments)
    assert preds.notna().all().all()


def test_gradients_flow(checkpoint):
    """Regression guard: posterior(X).mean.sum().backward() must propagate
    gradients into X. TabPFN's sklearn predict path uses ``inference_mode``
    which would silently break this — we call the architecture directly, so
    autograd should be alive end-to-end.
    """
    surrogate, _ = _fit_surrogate(checkpoint, "gaussian")
    X = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.double, requires_grad=True)
    post = surrogate.model.posterior(X)
    post.mean.sum().backward()
    assert X.grad is not None
    assert X.grad.abs().sum().item() > 0


def test_gaussian_vs_riemann_mean_agree(checkpoint):
    surrogate_g, experiments = _fit_surrogate(checkpoint, "gaussian")
    surrogate_r = surrogates.map(
        TabPFNDataModel(
            inputs=surrogate_g.inputs,
            outputs=surrogate_g.outputs,
            posterior_type="riemann",
            tabpfn_version="v2",
            checkpoint_path=checkpoint or None,
        )
    )
    surrogate_r.fit(experiments)
    pg = surrogate_g.predict(experiments)
    pr = surrogate_r.predict(experiments)
    # Both modes derive moments from the same bar-distribution logits, but the
    # bardist runs in float32 internally — ~1e-5 relative precision is the
    # expected ceiling.
    mean_col = [c for c in pg.columns if c.endswith("_pred")][0]
    rel_diff = (pg[mean_col] - pr[mean_col]).abs() / pg[mean_col].abs().clip(lower=1.0)
    assert rel_diff.max() < 1e-4
