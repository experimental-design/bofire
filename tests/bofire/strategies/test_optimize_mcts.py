"""Tests for optimize_mcts module."""

import random as stdlib_random

import pytest
import torch

import bofire.strategies.predictives.optimize_mcts as optimize_mcts_mod
from bofire.strategies.predictives.optimize_mcts import (
    MCTS,
    STOP,
    Categorical,
    Groups,
    NChooseK,
    Node,
    optimize_acqf_mcts,
)


# =============================================================================
# NChooseK tests
# =============================================================================


class TestNChooseK:
    def test_basic_construction(self):
        g = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        assert g.features == [0, 1, 2]
        assert g.min_count == 1
        assert g.max_count == 2

    def test_n_options(self):
        g = NChooseK(features=[0, 1, 2, 3], min_count=1, max_count=3)
        assert g.n_options == 4

    def test_n_features(self):
        g = NChooseK(features=[0, 2, 4], min_count=0, max_count=2)
        assert g.n_features == 3

    def test_non_contiguous_features(self):
        g = NChooseK(features=[0, 5, 10], min_count=1, max_count=2)
        assert g.n_features == 3
        assert g.n_options == 3

    # --- Validation ---

    def test_min_count_negative_raises(self):
        with pytest.raises(ValueError, match="Invalid NChooseK constraint"):
            NChooseK(features=[0, 1, 2], min_count=-1, max_count=2)

    def test_min_count_greater_than_max_count_raises(self):
        with pytest.raises(ValueError, match="Invalid NChooseK constraint"):
            NChooseK(features=[0, 1, 2], min_count=3, max_count=2)

    def test_max_count_greater_than_n_raises(self):
        with pytest.raises(ValueError, match="Invalid NChooseK constraint"):
            NChooseK(features=[0, 1], min_count=1, max_count=3)

    def test_min_count_equals_max_count_valid(self):
        g = NChooseK(features=[0, 1, 2], min_count=2, max_count=2)
        assert g.min_count == 2
        assert g.max_count == 2

    def test_zero_min_zero_max_valid(self):
        g = NChooseK(features=[0, 1], min_count=0, max_count=0)
        assert g.min_count == 0
        assert g.max_count == 0

    def test_empty_features_zero_counts(self):
        g = NChooseK(features=[], min_count=0, max_count=0)
        assert g.n_features == 0

    def test_empty_features_nonzero_counts_raises(self):
        with pytest.raises(ValueError, match="Invalid NChooseK constraint"):
            NChooseK(features=[], min_count=1, max_count=1)

    # --- legal_actions ---

    def test_legal_actions_empty_partial(self):
        g = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        actions = g.legal_actions(partial=(), stopped=False)
        # Can pick indices 0, 1, 2; cannot STOP yet (0 < min_count=1)
        assert actions == [0, 1, 2]

    def test_legal_actions_stop_available_after_min_count(self):
        g = NChooseK(features=[0, 1, 2], min_count=1, max_count=3)
        actions = g.legal_actions(partial=(0,), stopped=False)
        # Picked 1 item, min_count=1 met; can pick more (1, 2) or STOP
        assert 1 in actions
        assert 2 in actions
        assert STOP in actions

    def test_legal_actions_stop_not_available_below_min_count(self):
        g = NChooseK(features=[0, 1, 2, 3], min_count=2, max_count=3)
        actions = g.legal_actions(partial=(0,), stopped=False)
        # Only 1 picked, min_count=2 not met; STOP should NOT be available
        assert STOP not in actions

    def test_legal_actions_enforces_increasing_order(self):
        g = NChooseK(features=[0, 1, 2, 3, 4], min_count=1, max_count=5)
        actions = g.legal_actions(partial=(2,), stopped=False)
        # After picking index 2, only indices > 2 are legal (plus STOP)
        non_stop = [a for a in actions if a != STOP]
        assert all(a > 2 for a in non_stop)

    def test_legal_actions_when_stopped(self):
        g = NChooseK(features=[0, 1, 2], min_count=0, max_count=2)
        actions = g.legal_actions(partial=(0,), stopped=True)
        assert actions == []

    def test_legal_actions_at_max_count(self):
        g = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        actions = g.legal_actions(partial=(0, 1), stopped=False)
        assert actions == []

    def test_legal_actions_min_count_constrains_upper_bound(self):
        """When min_count requires more picks, high indices become unavailable
        because there wouldn't be enough remaining indices to satisfy min_count."""
        g = NChooseK(features=[0, 1, 2, 3], min_count=3, max_count=4)
        actions = g.legal_actions(partial=(), stopped=False)
        # Need at least 3 picks. After picking index i with 0 picks so far,
        # need 2 more from indices > i. So need n - (i+1) >= 2, i.e. i <= 1.
        # Legal: 0, 1 (not 2, 3 because not enough room for 3 total)
        assert actions == [0, 1]

    def test_legal_actions_min_equals_max(self):
        """When min_count == max_count, STOP is never available until max is reached."""
        g = NChooseK(features=[0, 1, 2], min_count=2, max_count=2)
        actions = g.legal_actions(partial=(), stopped=False)
        # Must pick exactly 2; first pick must leave room for 1 more
        assert STOP not in actions
        actions2 = g.legal_actions(partial=(0,), stopped=False)
        assert STOP not in actions2
        # After picking 2 items, max is reached
        actions3 = g.legal_actions(partial=(0, 1), stopped=False)
        assert actions3 == []

    def test_legal_actions_zero_min_count_stop_immediate(self):
        g = NChooseK(features=[0, 1, 2], min_count=0, max_count=2)
        actions = g.legal_actions(partial=(), stopped=False)
        # min_count=0, so STOP is immediately available
        assert STOP in actions
        assert 0 in actions

    def test_legal_actions_zero_max_count(self):
        g = NChooseK(features=[0, 1], min_count=0, max_count=0)
        actions = g.legal_actions(partial=(), stopped=False)
        # max_count=0 and len(partial)==0 means already at max; no actions available
        assert actions == []

    # --- is_complete ---

    def test_is_complete_not_yet(self):
        g = NChooseK(features=[0, 1, 2], min_count=1, max_count=3)
        assert g.is_complete(partial=(0,), stopped=False) is False

    def test_is_complete_when_stopped(self):
        g = NChooseK(features=[0, 1, 2], min_count=1, max_count=3)
        assert g.is_complete(partial=(0,), stopped=True) is True

    def test_is_complete_at_max_count(self):
        g = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        assert g.is_complete(partial=(0, 1), stopped=False) is True

    def test_is_complete_empty_partial_stopped(self):
        g = NChooseK(features=[0, 1, 2], min_count=0, max_count=2)
        assert g.is_complete(partial=(), stopped=True) is True

    def test_is_complete_zero_max(self):
        g = NChooseK(features=[0, 1], min_count=0, max_count=0)
        assert g.is_complete(partial=(), stopped=False) is True

    # --- frozen dataclass ---

    def test_frozen(self):
        g = NChooseK(features=[0, 1], min_count=0, max_count=1)
        with pytest.raises(AttributeError):
            g.min_count = 5  # type: ignore[misc]


# =============================================================================
# Categorical tests
# =============================================================================


class TestCategorical:
    def test_basic_construction(self):
        c = Categorical(dim=3, values=[0.0, 1.0, 2.0])
        assert c.dim == 3
        assert c.values == [0.0, 1.0, 2.0]

    def test_n_options(self):
        c = Categorical(dim=0, values=[10.0, 20.0, 30.0, 40.0])
        assert c.n_options == 4

    # --- Validation ---

    def test_single_value_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            Categorical(dim=0, values=[1.0])

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            Categorical(dim=0, values=[])

    def test_two_values_valid(self):
        c = Categorical(dim=0, values=[0.0, 1.0])
        assert c.n_options == 2

    # --- legal_actions ---

    def test_legal_actions_empty_partial(self):
        c = Categorical(dim=0, values=[0.0, 1.0, 2.0])
        actions = c.legal_actions(partial=(), stopped=False)
        assert actions == [0, 1, 2]

    def test_legal_actions_after_selection(self):
        c = Categorical(dim=0, values=[0.0, 1.0, 2.0])
        actions = c.legal_actions(partial=(1,), stopped=False)
        assert actions == []

    def test_legal_actions_ignores_stopped(self):
        """Categorical doesn't use the stopped flag, but should handle it."""
        c = Categorical(dim=0, values=[0.0, 1.0])
        actions = c.legal_actions(partial=(), stopped=True)
        # Still returns all options since stopped is not used for Categorical
        assert actions == [0, 1]

    # --- is_complete ---

    def test_is_complete_empty(self):
        c = Categorical(dim=0, values=[0.0, 1.0])
        assert c.is_complete(partial=(), stopped=False) is False

    def test_is_complete_after_selection(self):
        c = Categorical(dim=0, values=[0.0, 1.0])
        assert c.is_complete(partial=(0,), stopped=False) is True

    # --- frozen dataclass ---

    def test_frozen(self):
        c = Categorical(dim=0, values=[0.0, 1.0])
        with pytest.raises(AttributeError):
            c.dim = 5  # type: ignore[misc]


# =============================================================================
# Groups tests
# =============================================================================


class TestGroups:
    def _make_mixed_groups(self):
        nck1 = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        nck2 = NChooseK(features=[5, 6], min_count=1, max_count=2)
        cat1 = Categorical(dim=3, values=[0.0, 1.0])
        cat2 = Categorical(dim=4, values=[10.0, 20.0, 30.0])
        return Groups(groups=[nck1, nck2, cat1, cat2])

    def test_len(self):
        gs = self._make_mixed_groups()
        assert len(gs) == 4

    def test_len_empty(self):
        gs = Groups(groups=[])
        assert len(gs) == 0

    def test_nchooseks_property(self):
        gs = self._make_mixed_groups()
        ncks = gs.nchooseks
        assert len(ncks) == 2
        assert all(isinstance(g, NChooseK) for g in ncks)

    def test_categoricals_property(self):
        gs = self._make_mixed_groups()
        cats = gs.categoricals
        assert len(cats) == 2
        assert all(isinstance(g, Categorical) for g in cats)

    def test_all_nchoosek_features(self):
        gs = self._make_mixed_groups()
        feats = gs.all_nchoosek_features
        assert sorted(feats) == [0, 1, 2, 5, 6]

    def test_all_categorical_dims(self):
        gs = self._make_mixed_groups()
        dims = gs.all_categorical_dims
        assert dims == [3, 4]

    def test_only_nchoosek(self):
        nck = NChooseK(features=[0, 1], min_count=1, max_count=1)
        gs = Groups(groups=[nck])
        assert len(gs.nchooseks) == 1
        assert len(gs.categoricals) == 0
        assert gs.all_nchoosek_features == [0, 1]
        assert gs.all_categorical_dims == []

    def test_only_categorical(self):
        cat = Categorical(dim=7, values=[1.0, 2.0])
        gs = Groups(groups=[cat])
        assert len(gs.nchooseks) == 0
        assert len(gs.categoricals) == 1
        assert gs.all_nchoosek_features == []
        assert gs.all_categorical_dims == [7]

    def test_empty_groups(self):
        gs = Groups(groups=[])
        assert gs.nchooseks == []
        assert gs.categoricals == []
        assert gs.all_nchoosek_features == []
        assert gs.all_categorical_dims == []

    def test_feature_ordering_preserved(self):
        """all_nchoosek_features preserves order from groups list."""
        nck1 = NChooseK(features=[5, 3], min_count=1, max_count=1)
        nck2 = NChooseK(features=[1, 0], min_count=1, max_count=1)
        gs = Groups(groups=[nck1, nck2])
        assert gs.all_nchoosek_features == [5, 3, 1, 0]


# =============================================================================
# Node tests
# =============================================================================


class TestNode:
    """Tests for the MCTS Node dataclass."""

    @staticmethod
    def _two_group_groups() -> Groups:
        """Helper: two groups (1 NChooseK + 1 Categorical)."""
        nck = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        cat = Categorical(dim=3, values=[0.0, 1.0])
        return Groups(groups=[nck, cat])

    # --- Construction & defaults ---

    def test_default_values(self):
        node = Node(
            partial_by_group=((), ()),
            stopped_by_group=(False, False),
            group_idx=0,
        )
        assert node.n_visits == 0
        assert node.w_total == 0.0
        assert node.children == {}

    def test_explicit_values(self):
        node = Node(
            partial_by_group=((0, 1), (0,)),
            stopped_by_group=(True, False),
            group_idx=2,
            n_visits=10,
            w_total=5.5,
        )
        assert node.partial_by_group == ((0, 1), (0,))
        assert node.stopped_by_group == (True, False)
        assert node.group_idx == 2
        assert node.n_visits == 10
        assert node.w_total == 5.5

    # --- is_terminal ---

    def test_is_terminal_false_at_start(self):
        gs = self._two_group_groups()
        node = Node(
            partial_by_group=((), ()),
            stopped_by_group=(False, False),
            group_idx=0,
        )
        assert node.is_terminal(gs) is False

    def test_is_terminal_false_mid_group(self):
        gs = self._two_group_groups()
        node = Node(
            partial_by_group=((0,), ()),
            stopped_by_group=(False, False),
            group_idx=1,
        )
        assert node.is_terminal(gs) is False

    def test_is_terminal_true_past_last_group(self):
        gs = self._two_group_groups()
        node = Node(
            partial_by_group=((0,), (0,)),
            stopped_by_group=(True, False),
            group_idx=2,
        )
        assert node.is_terminal(gs) is True

    def test_is_terminal_true_when_group_idx_exceeds_len(self):
        gs = self._two_group_groups()
        node = Node(
            partial_by_group=((), ()),
            stopped_by_group=(False, False),
            group_idx=5,
        )
        assert node.is_terminal(gs) is True

    def test_is_terminal_empty_groups(self):
        gs = Groups(groups=[])
        node = Node(
            partial_by_group=(),
            stopped_by_group=(),
            group_idx=0,
        )
        assert node.is_terminal(gs) is True

    def test_is_terminal_single_group(self):
        gs = Groups(groups=[NChooseK(features=[0, 1], min_count=1, max_count=1)])
        node_before = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
        )
        node_after = Node(
            partial_by_group=((0,),),
            stopped_by_group=(False,),
            group_idx=1,
        )
        assert node_before.is_terminal(gs) is False
        assert node_after.is_terminal(gs) is True

    # --- mean_value ---

    def test_mean_value_zero_visits(self):
        node = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
        )
        assert node.mean_value() == 0.0

    def test_mean_value_with_visits(self):
        node = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
            n_visits=4,
            w_total=10.0,
        )
        assert node.mean_value() == pytest.approx(2.5)

    def test_mean_value_negative_reward(self):
        node = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
            n_visits=2,
            w_total=-6.0,
        )
        assert node.mean_value() == pytest.approx(-3.0)

    def test_mean_value_single_visit(self):
        node = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
            n_visits=1,
            w_total=7.0,
        )
        assert node.mean_value() == pytest.approx(7.0)

    # --- children dict ---

    def test_children_independent_per_node(self):
        """Each node gets its own children dict (not shared via mutable default)."""
        node_a = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
        )
        node_b = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
        )
        child = Node(
            partial_by_group=((0,),),
            stopped_by_group=(False,),
            group_idx=1,
        )
        node_a.children[0] = child
        assert 0 in node_a.children
        assert 0 not in node_b.children

    def test_children_keyed_by_action(self):
        parent = Node(
            partial_by_group=((), ()),
            stopped_by_group=(False, False),
            group_idx=0,
        )
        child_0 = Node(
            partial_by_group=((0,), ()),
            stopped_by_group=(False, False),
            group_idx=0,
        )
        child_stop = Node(
            partial_by_group=((), ()),
            stopped_by_group=(True, False),
            group_idx=1,
        )
        parent.children[0] = child_0
        parent.children[STOP] = child_stop
        assert parent.children[0] is child_0
        assert parent.children[STOP] is child_stop
        assert len(parent.children) == 2

    # --- mutability ---

    def test_mutable_n_visits_and_w_total(self):
        """Node is a regular (non-frozen) dataclass, so stats are mutable."""
        node = Node(
            partial_by_group=((),),
            stopped_by_group=(False,),
            group_idx=0,
        )
        node.n_visits += 1
        node.w_total += 3.5
        assert node.n_visits == 1
        assert node.w_total == 3.5


# =============================================================================
# MCTS tests
# =============================================================================


class TestMCTS:
    """Tests for the MCTS class."""

    # ---- helpers ----

    @staticmethod
    def _nck_only_groups() -> Groups:
        """Single NChooseK group: pick 1-2 from [0,1,2]."""
        return Groups(groups=[NChooseK(features=[0, 1, 2], min_count=1, max_count=2)])

    @staticmethod
    def _cat_only_groups() -> Groups:
        """Single Categorical group: dim 0, values [10, 20, 30]."""
        return Groups(groups=[Categorical(dim=0, values=[10.0, 20.0, 30.0])])

    @staticmethod
    def _mixed_groups() -> Groups:
        """NChooseK([0,1,2], 1, 2) + Categorical(dim=3, [0.0, 1.0])."""
        nck = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        cat = Categorical(dim=3, values=[0.0, 1.0])
        return Groups(groups=[nck, cat])

    @staticmethod
    def _constant_reward(value: float = 1.0):
        """Reward function that always returns a constant."""
        return lambda _feats, _cats: value

    # ---- __init__ ----

    def test_init_root_node(self):
        gs = self._mixed_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=0)
        assert mcts.root.group_idx == 0
        assert mcts.root.partial_by_group == ((), ())
        assert mcts.root.stopped_by_group == (False, False)
        assert mcts.root.n_visits == 0

    def test_init_defaults(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        assert mcts.c_uct == 0.01
        assert mcts.p_stop_rollout == 0.35
        assert mcts.pw_k0 == 2.0
        assert mcts.pw_alpha == 0.6
        assert mcts.best_value == float("-inf")
        assert mcts.best_selection is None

    def test_init_custom_params(self):
        gs = self._nck_only_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=self._constant_reward(),
            c_uct=2.0,
            p_stop_rollout=0.5,
            pw_k0=3.0,
            pw_alpha=0.7,
            seed=42,
        )
        assert mcts.c_uct == 2.0
        assert mcts.p_stop_rollout == 0.5
        assert mcts.pw_k0 == 3.0
        assert mcts.pw_alpha == 0.7

    # ---- _make_cache_key ----

    def test_cache_key_deterministic(self):
        gs = self._mixed_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        key1 = mcts._make_cache_key((0, 1), {3: 1.0})
        key2 = mcts._make_cache_key((0, 1), {3: 1.0})
        assert key1 == key2

    def test_cache_key_different_for_different_selections(self):
        gs = self._mixed_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        key_a = mcts._make_cache_key((0,), {3: 1.0})
        key_b = mcts._make_cache_key((1,), {3: 1.0})
        key_c = mcts._make_cache_key((0,), {3: 0.0})
        assert key_a != key_b
        assert key_a != key_c

    def test_cache_key_order_independent_for_cat_dict(self):
        """frozenset of dict items makes key independent of insertion order."""
        gs = Groups(
            groups=[
                Categorical(dim=0, values=[0.0, 1.0]),
                Categorical(dim=1, values=[2.0, 3.0]),
            ]
        )
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        cat_a = {0: 1.0, 1: 2.0}
        cat_b = {1: 2.0, 0: 1.0}
        assert mcts._make_cache_key((), cat_a) == mcts._make_cache_key((), cat_b)

    # ---- _cached_reward ----

    def test_cached_reward_calls_reward_fn_once(self):
        call_count = 0

        def counting_reward(_feats, _cats):
            nonlocal call_count
            call_count += 1
            return 5.0

        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=counting_reward)

        val1 = mcts._cached_reward((0,), {})
        val2 = mcts._cached_reward((0,), {})
        assert val1 == 5.0
        assert val2 == 5.0
        assert call_count == 1

    def test_cached_reward_different_keys_call_separately(self):
        call_count = 0

        def counting_reward(_feats, _cats):
            nonlocal call_count
            call_count += 1
            return float(call_count)

        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=counting_reward)

        mcts._cached_reward((0,), {})
        mcts._cached_reward((1,), {})
        assert call_count == 2

    def test_cache_stats_after_cached_reward(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        mcts._cached_reward((0,), {})
        mcts._cached_reward((0,), {})  # hit
        mcts._cached_reward((1,), {})  # miss
        stats = mcts.cache_stats()
        assert stats == {"hits": 1, "misses": 2, "size": 2}

    # ---- _child_limit (progressive widening) ----

    def test_child_limit_zero_visits(self):
        gs = self._nck_only_groups()
        mcts = MCTS(
            groups=gs, reward_fn=self._constant_reward(), pw_k0=2.0, pw_alpha=0.6
        )
        node = Node(
            partial_by_group=((),), stopped_by_group=(False,), group_idx=0, n_visits=0
        )
        # max(1, int(2.0 * max(1, 0)**0.6)) = max(1, int(2.0 * 1)) = 2
        assert mcts._child_limit(node) == 2

    def test_child_limit_increases_with_visits(self):
        gs = self._nck_only_groups()
        mcts = MCTS(
            groups=gs, reward_fn=self._constant_reward(), pw_k0=2.0, pw_alpha=0.6
        )
        limits = []
        for v in [1, 10, 100]:
            node = Node(
                partial_by_group=((),),
                stopped_by_group=(False,),
                group_idx=0,
                n_visits=v,
            )
            limits.append(mcts._child_limit(node))
        # Should be monotonically non-decreasing
        assert limits[0] <= limits[1] <= limits[2]
        # With more visits, limit should grow
        assert limits[2] > limits[0]

    # ---- _legal_actions ----

    def test_legal_actions_delegates_to_group(self):
        gs = self._nck_only_groups()  # features [0,1,2], min=1, max=2
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        node = Node(partial_by_group=((),), stopped_by_group=(False,), group_idx=0)
        actions = mcts._legal_actions(node)
        assert actions == [0, 1, 2]

    def test_legal_actions_terminal_returns_empty(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        node = Node(partial_by_group=((0,),), stopped_by_group=(False,), group_idx=1)
        assert mcts._legal_actions(node) == []

    # ---- _apply_action ----

    def test_apply_action_regular(self):
        gs = self._nck_only_groups()  # features [0,1,2], min=1, max=2
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        root = mcts.root  # group_idx=0, partial=((),)
        child = mcts._apply_action(root, 0)
        assert child.partial_by_group[0] == (0,)
        assert child.stopped_by_group[0] is False
        # min=1 met but max=2 not reached; group not complete, stays at group 0
        assert child.group_idx == 0

    def test_apply_action_completes_group(self):
        gs = self._nck_only_groups()  # max=2
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        # After picking 0, pick 1 => 2 picked == max_count, group complete
        node = Node(partial_by_group=((0,),), stopped_by_group=(False,), group_idx=0)
        child = mcts._apply_action(node, 1)
        assert child.partial_by_group[0] == (0, 1)
        assert child.group_idx == 1  # advanced past group 0

    def test_apply_action_stop(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        node = Node(partial_by_group=((0,),), stopped_by_group=(False,), group_idx=0)
        child = mcts._apply_action(node, STOP)
        assert child.stopped_by_group[0] is True
        assert child.partial_by_group[0] == (0,)  # unchanged
        assert child.group_idx == 1  # advanced

    def test_apply_action_categorical(self):
        gs = self._cat_only_groups()  # dim=0, values=[10,20,30]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        root = mcts.root
        child = mcts._apply_action(root, 2)  # pick index 2 (value 30)
        assert child.partial_by_group[0] == (2,)
        # Categorical is complete after 1 pick
        assert child.group_idx == 1

    def test_apply_action_does_not_mutate_parent(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        root = mcts.root
        original_partial = root.partial_by_group
        original_stopped = root.stopped_by_group
        mcts._apply_action(root, 0)
        assert root.partial_by_group == original_partial
        assert root.stopped_by_group == original_stopped

    def test_apply_action_mixed_groups_advances_correctly(self):
        gs = self._mixed_groups()  # NChooseK + Categorical
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        # Complete NChooseK by stopping after 1 pick
        node = Node(
            partial_by_group=((0,), ()),
            stopped_by_group=(False, False),
            group_idx=0,
        )
        after_stop = mcts._apply_action(node, STOP)
        assert after_stop.group_idx == 1  # now on categorical group
        # Pick categorical value
        after_cat = mcts._apply_action(after_stop, 0)
        assert after_cat.group_idx == 2  # past all groups (terminal)
        assert after_cat.partial_by_group == ((0,), (0,))

    # ---- _get_selection ----

    def test_get_selection_nchoosek_only(self):
        gs = self._nck_only_groups()  # features=[0,1,2]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        # Picked local indices 0 and 2 => features 0 and 2
        node = Node(partial_by_group=((0, 2),), stopped_by_group=(False,), group_idx=1)
        feats, cats = mcts._get_selection(node)
        assert feats == (0, 2)
        assert cats == {}

    def test_get_selection_cat_only(self):
        gs = self._cat_only_groups()  # dim=0, values=[10,20,30]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        node = Node(partial_by_group=((1,),), stopped_by_group=(False,), group_idx=1)
        feats, cats = mcts._get_selection(node)
        assert feats == ()
        assert cats == {0: 20.0}

    def test_get_selection_mixed(self):
        gs = self._mixed_groups()  # NChooseK features=[0,1,2], Cat dim=3 vals=[0,1]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        # NChooseK picked local 1 => feature 1; Cat picked local 1 => value 1.0
        node = Node(
            partial_by_group=((1,), (1,)),
            stopped_by_group=(True, False),
            group_idx=2,
        )
        feats, cats = mcts._get_selection(node)
        assert feats == (1,)
        assert cats == {3: 1.0}

    def test_get_selection_sorts_features(self):
        nck = NChooseK(features=[5, 1, 3], min_count=2, max_count=2)
        gs = Groups(groups=[nck])
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        # Picked local indices 0, 2 => features 5, 3 => sorted (3, 5)
        node = Node(partial_by_group=((0, 2),), stopped_by_group=(False,), group_idx=1)
        feats, _cats = mcts._get_selection(node)
        assert feats == (3, 5)

    def test_get_selection_empty_partial(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        # Stopped immediately with no picks (min_count=1 wouldn't normally allow
        # this, but _get_selection doesn't validate)
        node = Node(partial_by_group=((),), stopped_by_group=(True,), group_idx=1)
        feats, cats = mcts._get_selection(node)
        assert feats == ()
        assert cats == {}

    # ---- _rollout ----

    def test_rollout_reaches_terminal(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=42)
        feats, cats, _traj = mcts._rollout(mcts.root)
        # Should return a valid selection with at least min_count=1 features
        assert len(feats) >= 1
        assert all(f in [0, 1, 2] for f in feats)
        assert cats == {}

    def test_rollout_categorical_selects_one(self):
        gs = self._cat_only_groups()  # dim=0, values=[10,20,30]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=7)
        feats, cats, _traj = mcts._rollout(mcts.root)
        assert feats == ()
        assert 0 in cats
        assert cats[0] in [10.0, 20.0, 30.0]

    def test_rollout_mixed(self):
        gs = self._mixed_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=99)
        feats, cats, _traj = mcts._rollout(mcts.root)
        assert len(feats) >= 1
        assert all(f in [0, 1, 2] for f in feats)
        assert 3 in cats
        assert cats[3] in [0.0, 1.0]

    def test_rollout_deterministic_with_seed(self):
        gs = self._mixed_groups()
        results = []
        for _ in range(2):
            mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=123)
            results.append(mcts._rollout(mcts.root))
        assert results[0] == results[1]

    def test_rollout_does_not_mutate_input_node(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=0)
        root = mcts.root
        original_partial = root.partial_by_group
        original_stopped = root.stopped_by_group
        original_idx = root.group_idx
        mcts._rollout(root)
        assert root.partial_by_group == original_partial
        assert root.stopped_by_group == original_stopped
        assert root.group_idx == original_idx

    # ---- _backpropagate ----

    def test_backpropagate_updates_visits_and_reward(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        root = mcts.root
        child = Node(partial_by_group=((0,),), stopped_by_group=(False,), group_idx=0)
        path = [root, child]
        mcts._backpropagate(path, reward=3.0)
        assert root.n_visits == 1
        assert root.w_total == 3.0
        assert child.n_visits == 1
        assert child.w_total == 3.0

    def test_backpropagate_accumulates(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        root = mcts.root
        path = [root]
        mcts._backpropagate(path, reward=2.0)
        mcts._backpropagate(path, reward=5.0)
        assert root.n_visits == 2
        assert root.w_total == pytest.approx(7.0)

    # ---- _select_and_expand ----

    def test_select_and_expand_first_call_expands_root(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=0)
        leaf, path = mcts._select_and_expand()
        # Should have expanded one child from root
        assert len(path) == 2
        assert path[0] is mcts.root
        assert len(mcts.root.children) == 1

    def test_select_and_expand_grows_tree(self):
        gs = self._cat_only_groups()  # 3 options, single step per branch
        mcts = MCTS(
            groups=gs,
            reward_fn=self._constant_reward(),
            seed=0,
            pw_k0=10.0,  # large widening to allow many children
            pw_alpha=0.0,  # constant limit = max(1, int(10 * 1)) = 10
        )
        # Expand several times, backpropagating to allow further expansion
        for _ in range(3):
            leaf, path = mcts._select_and_expand()
            mcts._backpropagate(path, reward=1.0)
        # Root should have up to 3 children (one per expansion)
        assert len(mcts.root.children) >= 2

    # ---- run (integration) ----

    def test_run_returns_valid_selection_nchoosek(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(1.0), seed=42)
        feats, cats, val = mcts.run(n_iterations=20)
        assert len(feats) >= 1
        assert len(feats) <= 2
        assert all(f in [0, 1, 2] for f in feats)
        assert cats == {}
        assert val == 1.0

    def test_run_returns_valid_selection_categorical(self):
        gs = self._cat_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(2.0), seed=42)
        feats, cats, val = mcts.run(n_iterations=20)
        assert feats == ()
        assert 0 in cats
        assert cats[0] in [10.0, 20.0, 30.0]
        assert val == 2.0

    def test_run_returns_valid_selection_mixed(self):
        gs = self._mixed_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(3.0), seed=42)
        feats, cats, val = mcts.run(n_iterations=30)
        assert 1 <= len(feats) <= 2
        assert all(f in [0, 1, 2] for f in feats)
        assert 3 in cats
        assert cats[3] in [0.0, 1.0]
        assert val == 3.0

    def test_run_finds_best_reward(self):
        """MCTS should track the best reward across iterations."""
        gs = self._nck_only_groups()

        def reward_by_count(feats, _cats):
            return float(len(feats))

        mcts = MCTS(groups=gs, reward_fn=reward_by_count, seed=42)
        feats, _cats, val = mcts.run(n_iterations=50)
        # Best possible: pick 2 features (max_count=2) => reward 2.0
        assert val == 2.0
        assert len(feats) == 2

    def test_run_deterministic_with_seed(self):
        gs = self._mixed_groups()
        results = []
        for _ in range(2):
            mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=77)
            results.append(mcts.run(n_iterations=30))
        assert results[0] == results[1]

    def test_run_updates_root_visits(self):
        gs = self._nck_only_groups()
        n_iter = 25
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=0)
        mcts.run(n_iterations=n_iter)
        # Root visits equals the number of novel (non-cached) evaluations,
        # which is at most n_iter (cache hits skip backpropagation).
        assert 0 < mcts.root.n_visits <= n_iter

    def test_run_populates_cache(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=0)
        mcts.run(n_iterations=30)
        stats = mcts.cache_stats()
        assert stats["size"] > 0
        assert stats["misses"] > 0
        # With repeated selections, should have some hits
        assert stats["hits"] + stats["misses"] == 30

    def test_run_zero_iterations(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=0)
        feats, cats, val = mcts.run(n_iterations=0)
        assert feats == ()
        assert cats == {}
        assert val == float("-inf")

    def test_run_prefers_higher_reward(self):
        """Given a reward function that varies by selection, MCTS should prefer better ones."""
        nck = NChooseK(features=[0, 1, 2], min_count=1, max_count=1)
        gs = Groups(groups=[nck])

        def reward_fn(feats, _cats):
            # Feature 2 gives the best reward
            if 2 in feats:
                return 10.0
            if 1 in feats:
                return 5.0
            return 1.0

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=42)
        feats, _cats, val = mcts.run(n_iterations=100)
        # With enough iterations, MCTS should find the best
        assert val == 10.0
        assert 2 in feats

    # ---- cache_stats ----

    def test_cache_stats_initial(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        assert mcts.cache_stats() == {"hits": 0, "misses": 0, "size": 0}


# =============================================================================
# Adaptive p_stop tests
# =============================================================================


class TestAdaptivePStop:
    """Tests for adaptive per-group p_stop_rollout."""

    @staticmethod
    def _single_nck_groups():
        """Single NChooseK group: 5 features, pick 1-4."""
        nck = NChooseK(features=[0, 1, 2, 3, 4], min_count=1, max_count=4)
        return Groups(groups=[nck])

    @staticmethod
    def _two_nck_groups():
        """Two NChooseK groups for multi-group tests."""
        g1 = NChooseK(features=[0, 1, 2], min_count=1, max_count=3)
        g2 = NChooseK(features=[10, 11, 12], min_count=1, max_count=3)
        return Groups(groups=[g1, g2])

    def test_adaptive_p_stop_disabled_returns_fixed(self):
        """When adaptive_p_stop=False, _compute_adaptive_p_stop returns p_stop_rollout."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=False,
            p_stop_rollout=0.42,
            seed=0,
        )
        # Even after populating stats, should return fixed value
        mcts.cardinality_stats[(0, 2)] = (10, 50.0)
        mcts.cardinality_stats[(0, 3)] = (10, 30.0)
        mcts.reward_min = 0.0
        mcts.reward_max = 10.0
        mcts.group_rollout_counts[0] = 100
        assert mcts._compute_adaptive_p_stop(0, 2) == 0.42

    def test_adaptive_p_stop_no_data_returns_prior(self):
        """With no cardinality stats, returns p_stop_rollout."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            p_stop_rollout=0.35,
            seed=0,
        )
        assert mcts._compute_adaptive_p_stop(0, 2) == 0.35

    def test_adaptive_p_stop_stop_better(self):
        """When stopping gives higher reward than continuing, p should be > 0.5."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            p_stop_rollout=0.35,
            p_stop_warmup=1,  # minimal warmup
            seed=0,
        )
        # Stopping at cardinality 2 is much better than continuing to 3 or 4
        mcts.cardinality_stats[(0, 2)] = (50, 500.0)  # mean=10.0
        mcts.cardinality_stats[(0, 3)] = (50, 100.0)  # mean=2.0
        mcts.cardinality_stats[(0, 4)] = (50, 50.0)  # mean=1.0
        mcts.reward_min = 0.0
        mcts.reward_max = 15.0
        mcts.group_rollout_counts[0] = 100
        p = mcts._compute_adaptive_p_stop(0, 2)
        assert p > 0.5

    def test_adaptive_p_stop_continue_better(self):
        """When continuing gives higher reward, p should be < 0.5."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            p_stop_rollout=0.35,
            p_stop_warmup=1,
            seed=0,
        )
        # Stopping at cardinality 2 is much worse than continuing to 3
        mcts.cardinality_stats[(0, 2)] = (50, 100.0)  # mean=2.0
        mcts.cardinality_stats[(0, 3)] = (50, 500.0)  # mean=10.0
        mcts.reward_min = 0.0
        mcts.reward_max = 15.0
        mcts.group_rollout_counts[0] = 100
        p = mcts._compute_adaptive_p_stop(0, 2)
        assert p < 0.5

    def test_adaptive_p_stop_warmup_blending(self):
        """During warmup, result is blended between prior and learned."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            p_stop_rollout=0.35,
            p_stop_warmup=100,
            seed=0,
        )
        # Stop much better than continue → p_learned should be high
        mcts.cardinality_stats[(0, 2)] = (50, 500.0)  # mean=10.0
        mcts.cardinality_stats[(0, 3)] = (50, 50.0)  # mean=1.0
        mcts.reward_min = 0.0
        mcts.reward_max = 15.0

        # At 10 visits out of 100 warmup, alpha = 0.1
        mcts.group_rollout_counts[0] = 10
        p_early = mcts._compute_adaptive_p_stop(0, 2)

        # At 100 visits, alpha = 1.0 (fully learned)
        mcts.group_rollout_counts[0] = 100
        p_late = mcts._compute_adaptive_p_stop(0, 2)

        # Early should be closer to prior (0.35), late should be more extreme
        assert abs(p_early - 0.35) < abs(p_late - 0.35)
        # Both should be > 0.35 since stopping is better
        assert p_early > 0.35
        assert p_late > p_early

    def test_adaptive_p_stop_uses_max_over_higher(self):
        """E_continue should use the max over higher cardinalities, not average."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            p_stop_rollout=0.35,
            p_stop_warmup=1,
            seed=0,
        )
        # Stopping at 2 gives mean=5.0
        # Continuing to 3 gives mean=3.0, but continuing to 4 gives mean=8.0
        # So E_continue should be 8.0 (max), making continue look better
        mcts.cardinality_stats[(0, 2)] = (50, 250.0)  # mean=5.0
        mcts.cardinality_stats[(0, 3)] = (50, 150.0)  # mean=3.0
        mcts.cardinality_stats[(0, 4)] = (50, 400.0)  # mean=8.0
        mcts.reward_min = 0.0
        mcts.reward_max = 10.0
        mcts.group_rollout_counts[0] = 100
        p = mcts._compute_adaptive_p_stop(0, 2)
        # Since E_continue=8.0 > E_stop=5.0, should favor continuing → p < 0.5
        assert p < 0.5

    def test_adaptive_p_stop_no_continue_data_returns_prior(self):
        """With stop data but no higher-cardinality data, returns prior."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            p_stop_rollout=0.35,
            p_stop_warmup=1,
            seed=0,
        )
        # Only have data for stopping at cardinality 4 (max), no higher possible
        mcts.cardinality_stats[(0, 4)] = (50, 250.0)
        mcts.reward_min = 0.0
        mcts.reward_max = 10.0
        mcts.group_rollout_counts[0] = 100
        assert mcts._compute_adaptive_p_stop(0, 4) == 0.35

    def test_adaptive_p_stop_zero_reward_range_returns_prior(self):
        """When reward range is zero, returns prior."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            p_stop_rollout=0.35,
            seed=0,
        )
        mcts.cardinality_stats[(0, 2)] = (50, 250.0)
        mcts.cardinality_stats[(0, 3)] = (50, 250.0)
        mcts.reward_min = 5.0
        mcts.reward_max = 5.0  # zero range
        mcts.group_rollout_counts[0] = 100
        assert mcts._compute_adaptive_p_stop(0, 2) == 0.35

    def test_update_cardinality_stats_basic(self):
        """_update_cardinality_stats correctly updates for a single group."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            seed=0,
        )
        # Features [0, 1, 2, 3, 4], pick 2 features: cardinality = 2
        mcts._update_cardinality_stats(10.0, (0, 3))
        assert mcts.cardinality_stats[(0, 2)] == (1, 10.0)
        assert mcts.group_rollout_counts[0] == 1

        # Another update with different cardinality
        mcts._update_cardinality_stats(5.0, (1,))
        assert mcts.cardinality_stats[(0, 1)] == (1, 5.0)
        assert mcts.group_rollout_counts[0] == 2

    def test_update_cardinality_stats_multiple_groups(self):
        """_update_cardinality_stats correctly handles multiple NChooseK groups."""
        gs = self._two_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            seed=0,
        )
        # Group 0 features=[0,1,2], Group 1 features=[10,11,12]
        # Select 0 and 1 from group 0 (cardinality=2), 10 from group 1 (cardinality=1)
        mcts._update_cardinality_stats(7.0, (0, 1, 10))
        assert mcts.cardinality_stats[(0, 2)] == (1, 7.0)
        assert mcts.cardinality_stats[(1, 1)] == (1, 7.0)
        assert mcts.group_rollout_counts[0] == 1
        assert mcts.group_rollout_counts[1] == 1

    def test_update_cardinality_stats_accumulates(self):
        """Stats accumulate across multiple updates."""
        gs = self._single_nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 1.0,
            adaptive_p_stop=True,
            seed=0,
        )
        mcts._update_cardinality_stats(10.0, (0, 3))
        mcts._update_cardinality_stats(6.0, (1, 4))
        assert mcts.cardinality_stats[(0, 2)] == (2, 16.0)
        assert mcts.group_rollout_counts[0] == 2

    def test_run_with_adaptive_populates_stats(self):
        """Running MCTS with adaptive_p_stop=True populates cardinality_stats."""
        gs = self._single_nck_groups()

        def reward_fn(feats, _cats):
            return float(len(feats))

        mcts = MCTS(
            groups=gs,
            reward_fn=reward_fn,
            adaptive_p_stop=True,
            seed=42,
        )
        mcts.run(n_iterations=50)
        # Should have accumulated stats for at least some cardinalities
        assert len(mcts.cardinality_stats) > 0
        assert mcts.group_rollout_counts[0] > 0
        assert mcts.reward_min < mcts.reward_max

    def test_run_without_adaptive_leaves_stats_empty(self):
        """Running MCTS with adaptive_p_stop=False does not populate stats."""
        gs = self._single_nck_groups()

        def reward_fn(feats, _cats):
            return float(len(feats))

        mcts = MCTS(
            groups=gs,
            reward_fn=reward_fn,
            adaptive_p_stop=False,
            seed=42,
        )
        mcts.run(n_iterations=50)
        assert len(mcts.cardinality_stats) == 0
        assert all(c == 0 for c in mcts.group_rollout_counts)

    def test_adaptive_p_stop_temperature_effect(self):
        """Lower temperature makes the sigmoid more decisive."""
        gs = self._single_nck_groups()
        base_kwargs = {
            "groups": gs,
            "reward_fn": lambda f, c: 1.0,
            "adaptive_p_stop": True,
            "p_stop_rollout": 0.35,
            "p_stop_warmup": 1,
            "seed": 0,
        }

        def setup_stats(mcts):
            mcts.cardinality_stats[(0, 2)] = (50, 400.0)  # mean=8.0
            mcts.cardinality_stats[(0, 3)] = (50, 200.0)  # mean=4.0
            mcts.reward_min = 0.0
            mcts.reward_max = 10.0
            mcts.group_rollout_counts[0] = 100

        mcts_low_temp = MCTS(**base_kwargs, p_stop_temperature=0.1)
        setup_stats(mcts_low_temp)
        p_low = mcts_low_temp._compute_adaptive_p_stop(0, 2)

        mcts_high_temp = MCTS(**base_kwargs, p_stop_temperature=1.0)
        setup_stats(mcts_high_temp)
        p_high = mcts_high_temp._compute_adaptive_p_stop(0, 2)

        # Both should be > 0.5 (stopping is better)
        assert p_low > 0.5
        assert p_high > 0.5
        # Low temp → more extreme (closer to 1.0)
        assert p_low > p_high


# =============================================================================
# Reward normalization tests
# =============================================================================


class TestRewardNormalization:
    """Tests for the MCTS reward normalization feature."""

    @staticmethod
    def _simple_groups() -> Groups:
        """Single NChooseK group: pick 1-2 from [0,1,2]."""
        return Groups(groups=[NChooseK(features=[0, 1, 2], min_count=1, max_count=2)])

    def test_normalize_reward_basic(self):
        """Verify [0, 1] mapping after seeing min and max rewards."""
        groups = self._simple_groups()
        mcts = MCTS(groups=groups, reward_fn=lambda f, c: 0.0, seed=0)
        mcts.reward_min = 10.0
        mcts.reward_max = 110.0
        assert mcts._normalize_reward(10.0) == pytest.approx(0.0)
        assert mcts._normalize_reward(110.0) == pytest.approx(1.0)
        assert mcts._normalize_reward(60.0) == pytest.approx(0.5)

    def test_normalize_reward_zero_range(self):
        """Returns 0.5 when all rewards are identical (range is zero)."""
        groups = self._simple_groups()
        mcts = MCTS(groups=groups, reward_fn=lambda f, c: 0.0, seed=0)
        mcts.reward_min = 42.0
        mcts.reward_max = 42.0
        assert mcts._normalize_reward(42.0) == pytest.approx(0.5)

    def test_normalize_reward_negative_rewards(self):
        """Works correctly with negative reward ranges."""
        groups = self._simple_groups()
        mcts = MCTS(groups=groups, reward_fn=lambda f, c: 0.0, seed=0)
        mcts.reward_min = -100.0
        mcts.reward_max = -10.0
        assert mcts._normalize_reward(-100.0) == pytest.approx(0.0)
        assert mcts._normalize_reward(-10.0) == pytest.approx(1.0)
        assert mcts._normalize_reward(-55.0) == pytest.approx(0.5)

    def test_normalization_disabled_backpropagates_raw(self):
        """When normalize_rewards=False, w_total accumulates raw rewards."""
        groups = self._simple_groups()
        rewards = iter([50.0, 100.0])
        mcts = MCTS(
            groups=groups,
            reward_fn=lambda f, c: next(rewards),
            normalize_rewards=False,
            seed=0,
        )
        mcts.run(n_iterations=2)
        # Root should have accumulated raw rewards
        assert mcts.root.w_total == pytest.approx(50.0 + 100.0)

    def test_normalization_enabled_backpropagates_normalized(self):
        """When normalize_rewards=True, w_total accumulates normalized rewards in [0, 1]."""
        groups = self._simple_groups()
        call_count = 0
        reward_values = [50.0, 100.0, 75.0]

        def reward_fn(f, c):
            nonlocal call_count
            idx = min(call_count, len(reward_values) - 1)
            call_count += 1
            return reward_values[idx]

        mcts = MCTS(
            groups=groups,
            reward_fn=reward_fn,
            normalize_rewards=True,
            seed=0,
        )
        mcts.run(n_iterations=3)

        # All w_total values in the tree should be <= n_visits (since max normalized = 1.0)
        def check_node(node):
            if node.n_visits > 0:
                mean = node.w_total / node.n_visits
                assert (
                    0.0 <= mean <= 1.0 + 1e-9
                ), f"mean_value={mean} out of [0,1] range"
            for child in node.children.values():
                check_node(child)

        check_node(mcts.root)

    def test_best_value_stays_raw(self):
        """best_value must reflect raw reward regardless of normalization flag."""
        groups = self._simple_groups()
        raw_reward = 9999.0
        mcts = MCTS(
            groups=groups,
            reward_fn=lambda f, c: raw_reward,
            normalize_rewards=True,
            seed=0,
        )
        mcts.run(n_iterations=5)
        assert mcts.best_value == pytest.approx(raw_reward)

    def test_run_with_normalization_finds_optimum(self):
        """MCTS with normalization still finds the optimum on a simple problem."""
        g = NChooseK(features=[0, 1, 2, 3, 4], min_count=2, max_count=2)
        groups = Groups(groups=[g])
        target = frozenset({1, 3})

        def reward_fn(feats, _cats):
            return 100.0 if set(feats) == target else 0.0

        mcts = MCTS(
            groups=groups,
            reward_fn=reward_fn,
            normalize_rewards=True,
            seed=42,
        )
        mcts.run(n_iterations=200)
        assert mcts.best_value == pytest.approx(100.0)

    def test_virtual_loss_with_shifted_rewards(self):
        """Normalization fixes virtual loss dilution on shifted/negative rewards.

        Without normalization, virtual loss (adding 0 reward on cache hit) dilutes
        toward 0, which is wrong for shifted rewards. With normalization, the
        backpropagated values are in [0, 1], so virtual loss dilutes toward 0
        which is the minimum normalized reward — the correct behavior.
        """
        g = NChooseK(features=[0, 1, 2, 3], min_count=1, max_count=2)
        groups = Groups(groups=[g])

        # All rewards are large positive values; virtual loss of 0 would be
        # extremely penalizing without normalization
        def reward_fn(feats, _cats):
            if set(feats) == {0, 2}:
                return 1000.0
            return 900.0 + len(feats)

        # With normalization: virtual loss dilutes toward 0 (= worst normalized)
        mcts_norm = MCTS(
            groups=groups,
            reward_fn=reward_fn,
            normalize_rewards=True,
            seed=0,
        )
        mcts_norm.run(n_iterations=150)

        # Without normalization: virtual loss dilutes toward 0 (far below 900)
        mcts_raw = MCTS(
            groups=groups,
            reward_fn=reward_fn,
            normalize_rewards=False,
            seed=0,
        )
        mcts_raw.run(n_iterations=150)

        # Both should find the optimum given enough iterations
        assert mcts_norm.best_value == pytest.approx(1000.0)
        assert mcts_raw.best_value == pytest.approx(1000.0)

    def test_normalize_rewards_default_is_true(self):
        """The default value for normalize_rewards should be True."""
        groups = self._simple_groups()
        mcts = MCTS(groups=groups, reward_fn=lambda f, c: 0.0, seed=0)
        assert mcts.normalize_rewards is True


# =============================================================================
# Rollout policy tests
# =============================================================================


class TestRolloutPolicy:
    """Tests for the blended softmax rollout policy."""

    @staticmethod
    def _nck_groups() -> Groups:
        """Single NChooseK group: 5 features, pick 1-3."""
        return Groups(
            groups=[NChooseK(features=[0, 1, 2, 3, 4], min_count=1, max_count=3)]
        )

    @staticmethod
    def _mixed_groups() -> Groups:
        """NChooseK([0,1,2], 1, 2) + Categorical(dim=3, [0.0, 1.0])."""
        nck = NChooseK(features=[0, 1, 2], min_count=1, max_count=2)
        cat = Categorical(dim=3, values=[0.0, 1.0])
        return Groups(groups=[nck, cat])

    # ---- Scoring tests ----

    def test_score_rollout_actions_no_stats(self):
        """With no stats, all scores equal novelty_weight."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_novelty_weight=2.0,
            seed=0,
        )
        scores = mcts._score_rollout_actions(0, [0, 1, 2, STOP])
        for a in [0, 1, 2, STOP]:
            assert scores[a] == pytest.approx(2.0)

    def test_score_rollout_actions_with_stats(self):
        """Mean + novelty bonus computed correctly."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_novelty_weight=1.0,
            seed=0,
        )
        # Seed stats: action 0 visited 3 times with total reward 9.0
        mcts.rollout_stats[(0, 0)] = (3, 9.0)
        scores = mcts._score_rollout_actions(0, [0, 1])
        # Action 0: mean=3.0, novelty=1/sqrt(4)=0.5
        assert scores[0] == pytest.approx(3.0 + 0.5)
        # Action 1: no stats, score = novelty_weight = 1.0
        assert scores[1] == pytest.approx(1.0)

    def test_score_rollout_actions_includes_stop(self):
        """STOP is scored like any other action."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_novelty_weight=1.0,
            seed=0,
        )
        mcts.rollout_stats[(0, STOP)] = (4, 20.0)
        scores = mcts._score_rollout_actions(0, [0, STOP])
        # STOP: mean=5.0, novelty=1/sqrt(5)
        import math

        assert scores[STOP] == pytest.approx(5.0 + 1.0 / math.sqrt(5))
        # Action 0: no stats
        assert scores[0] == pytest.approx(1.0)

    # ---- Sampling tests ----

    def test_sample_rollout_action_returns_legal(self):
        """Returned action is in legal_actions."""
        gs = self._nck_groups()
        mcts = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=42)
        legal = [0, 1, 2, STOP]
        for _ in range(20):
            action = mcts._sample_rollout_action(0, legal)
            assert action in legal

    def test_sample_rollout_action_deterministic_seed(self):
        """Same seed produces same action."""
        gs = self._nck_groups()
        actions = []
        for _ in range(2):
            mcts = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=77)
            actions.append(mcts._sample_rollout_action(0, [0, 1, 2]))
        assert actions[0] == actions[1]

    def test_sample_rollout_action_respects_epsilon(self):
        """With epsilon=1.0, distribution is uniform."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_epsilon=1.0,
            seed=0,
        )
        # Seed stats to make action 0 strongly preferred by policy
        mcts.rollout_stats[(0, 0)] = (100, 10000.0)
        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(3000):
            a = mcts._sample_rollout_action(0, [0, 1, 2])
            counts[a] += 1
        # With epsilon=1.0, should be roughly uniform (each ~1000)
        for a in [0, 1, 2]:
            assert counts[a] > 700, f"Action {a} count {counts[a]} too low for uniform"

    def test_sample_rollout_action_respects_tau(self):
        """Low tau concentrates on the best action."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_epsilon=0.0,
            rollout_tau=0.01,  # Very low temperature
            seed=0,
        )
        # Action 0 is much better
        mcts.rollout_stats[(0, 0)] = (50, 500.0)  # mean = 10.0
        mcts.rollout_stats[(0, 1)] = (50, 50.0)  # mean = 1.0
        mcts.rollout_stats[(0, 2)] = (50, 50.0)  # mean = 1.0
        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(1000):
            a = mcts._sample_rollout_action(0, [0, 1, 2])
            counts[a] += 1
        # Action 0 should dominate
        assert (
            counts[0] > 900
        ), f"Action 0 should dominate with low tau, got {counts[0]}"

    # ---- Stats update tests ----

    def test_update_rollout_stats_basic(self):
        """Trajectory updates dict correctly."""
        gs = self._nck_groups()
        mcts = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=0)
        trajectory = [(0, 1), (0, 2)]
        mcts._update_rollout_stats(trajectory, 5.0)
        assert mcts.rollout_stats[(0, 1)] == (1, 5.0)
        assert mcts.rollout_stats[(0, 2)] == (1, 5.0)

    def test_update_rollout_stats_accumulates(self):
        """Multiple updates accumulate."""
        gs = self._nck_groups()
        mcts = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=0)
        mcts._update_rollout_stats([(0, 1)], 3.0)
        mcts._update_rollout_stats([(0, 1)], 7.0)
        assert mcts.rollout_stats[(0, 1)] == (2, 10.0)

    def test_update_rollout_stats_empty_trajectory(self):
        """No-op for empty trajectory."""
        gs = self._nck_groups()
        mcts = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=0)
        mcts._update_rollout_stats([], 5.0)
        assert mcts.rollout_stats == {}

    # ---- Rollout integration tests ----

    def test_rollout_returns_three_tuple(self):
        """Rollout always returns (feats, cats, trajectory)."""
        gs = self._mixed_groups()
        mcts = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=42)
        result = mcts._rollout(mcts.root)
        assert len(result) == 3
        feats, cats, trajectory = result
        assert isinstance(feats, tuple)
        assert isinstance(cats, dict)
        assert isinstance(trajectory, list)

    def test_rollout_trajectory_records_actions(self):
        """Trajectory is non-empty for non-trivial groups."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_policy=True,
            seed=42,
        )
        _feats, _cats, trajectory = mcts._rollout(mcts.root)
        assert len(trajectory) > 0
        # All entries are (group_idx, action) pairs
        for g, a in trajectory:
            assert isinstance(g, int)
            assert isinstance(a, int)

    def test_rollout_policy_disabled_still_records_trajectory(self):
        """Trajectory collected even when rollout_policy=False."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_policy=False,
            seed=42,
        )
        _feats, _cats, trajectory = mcts._rollout(mcts.root)
        assert len(trajectory) > 0

    def test_rollout_policy_uses_learned_stats(self):
        """After seeding stats, policy biases toward good actions."""
        gs = Groups(groups=[NChooseK(features=[0, 1, 2], min_count=1, max_count=1)])
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: 0.0,
            rollout_policy=True,
            rollout_epsilon=0.0,
            rollout_tau=0.01,
            seed=0,
        )
        # Make action 2 strongly preferred
        mcts.rollout_stats[(0, 2)] = (100, 1000.0)  # mean = 10.0
        mcts.rollout_stats[(0, 0)] = (100, 100.0)  # mean = 1.0
        mcts.rollout_stats[(0, 1)] = (100, 100.0)  # mean = 1.0

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(500):
            mcts_i = MCTS(
                groups=gs,
                reward_fn=lambda f, c: 0.0,
                rollout_policy=True,
                rollout_epsilon=0.0,
                rollout_tau=0.01,
                seed=None,
            )
            mcts_i.rollout_stats = dict(mcts.rollout_stats)
            feats, _cats, _traj = mcts_i._rollout(mcts_i.root)
            for f in feats:
                counts[f] += 1
        # Action 2 (feature 2) should be picked most often
        assert counts[2] > counts[0]
        assert counts[2] > counts[1]

    # ---- End-to-end tests ----

    def test_run_with_rollout_policy_converges(self):
        """Convergence on needle problem with rollout policy enabled."""
        g = NChooseK(features=list(range(10)), min_count=2, max_count=3)
        gs = Groups(groups=[g])
        target = {3, 7}

        def reward_fn(feats, _cats):
            feat_set = set(feats)
            if feat_set == target:
                return 100.0
            overlap = len(feat_set & target)
            extras = len(feat_set - target)
            return overlap * 20.0 - extras * 5.0

        mcts = MCTS(
            groups=gs,
            reward_fn=reward_fn,
            rollout_policy=True,
            rollout_epsilon=0.3,
            rollout_tau=1.0,
            seed=42,
        )
        _feats, _cats, val = mcts.run(n_iterations=300)
        assert val == 100.0

    def test_rollout_stats_populated_after_run(self):
        """rollout_stats is non-empty after run()."""
        gs = self._nck_groups()
        mcts = MCTS(
            groups=gs,
            reward_fn=lambda f, c: float(len(f)),
            rollout_policy=True,
            seed=0,
        )
        mcts.run(n_iterations=50)
        assert len(mcts.rollout_stats) > 0

    def test_run_without_rollout_policy_unchanged(self):
        """With rollout_policy=False, results match old behavior (seed-deterministic)."""
        gs = Groups(groups=[NChooseK(features=[0, 1, 2], min_count=1, max_count=2)])

        def reward_fn(feats, _cats):
            return float(len(feats)) * 10.0

        # Run with rollout_policy=False (default)
        mcts1 = MCTS(groups=gs, reward_fn=reward_fn, rollout_policy=False, seed=42)
        _f1, _c1, val1 = mcts1.run(n_iterations=50)

        # Run again with same seed
        mcts2 = MCTS(groups=gs, reward_fn=reward_fn, rollout_policy=False, seed=42)
        _f2, _c2, val2 = mcts2.run(n_iterations=50)

        assert val1 == val2

    def test_rollout_policy_default_is_true(self):
        """The default value for rollout_policy should be True."""
        gs = self._nck_groups()
        mcts = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=0)
        assert mcts.rollout_policy is True


# =============================================================================
# MCTS convergence / integration tests
# =============================================================================


class TestMCTSConvergence:
    """Integration tests verifying that MCTS converges to known optima."""

    def test_nchoosek_single_optimum(self):
        """5-choose-2 problem where exactly one pair is optimal.

        Features [0..4], pick exactly 2.  Reward = 100 iff {1, 3} selected,
        else 0.  MCTS must find this needle.
        """
        nck = NChooseK(features=[0, 1, 2, 3, 4], min_count=2, max_count=2)
        gs = Groups(groups=[nck])

        def reward_fn(feats, _cats):
            return 100.0 if set(feats) == {1, 3} else 0.0

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=0)
        feats, cats, val = mcts.run(n_iterations=200)
        assert val == 100.0
        assert set(feats) == {1, 3}
        assert cats == {}

    def test_categorical_single_optimum(self):
        """Two categorical dimensions; only one combination is optimal."""
        cat1 = Categorical(dim=0, values=[0.0, 1.0, 2.0])
        cat2 = Categorical(dim=1, values=[10.0, 20.0])
        gs = Groups(groups=[cat1, cat2])

        def reward_fn(_feats, cats):
            if cats.get(0) == 2.0 and cats.get(1) == 10.0:
                return 50.0
            return 1.0

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=42)
        feats, cats, val = mcts.run(n_iterations=100)
        assert val == 50.0
        assert cats == {0: 2.0, 1: 10.0}
        assert feats == ()

    def test_mixed_nchoosek_and_categorical_optimum(self):
        """Joint NChooseK + Categorical problem with a single optimum."""
        nck = NChooseK(features=[0, 1, 2, 3], min_count=1, max_count=2)
        cat = Categorical(dim=5, values=[0.0, 1.0, 2.0])
        gs = Groups(groups=[nck, cat])

        def reward_fn(feats, cats):
            # Optimum: select features {0, 3} and categorical dim 5 = 2.0
            if set(feats) == {0, 3} and cats.get(5) == 2.0:
                return 100.0
            return 1.0

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=7)
        feats, cats, val = mcts.run(n_iterations=300)
        assert val == 100.0
        assert set(feats) == {0, 3}
        assert cats == {5: 2.0}

    def test_best_value_improves_over_iterations(self):
        """Run MCTS in stages and verify the best value never decreases."""
        nck = NChooseK(features=[0, 1, 2, 3, 4], min_count=1, max_count=3)
        gs = Groups(groups=[nck])

        def reward_fn(feats, _cats):
            # Best: {1, 2, 4} => reward 100
            if set(feats) == {1, 2, 4}:
                return 100.0
            return float(len(feats))

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=42)

        best_values = []
        for _ in range(5):
            mcts.run(n_iterations=40)
            best_values.append(mcts.best_value)

        # best_value must be monotonically non-decreasing across stages
        for i in range(1, len(best_values)):
            assert best_values[i] >= best_values[i - 1]
        # Should find the optimum within 200 total iterations
        assert best_values[-1] == 100.0

    def test_variable_count_optimum(self):
        """Optimum requires a specific count of features (not max).

        Pick 1-4 from [0..5]. Reward is 80 for exactly {2, 5}, and
        len(feats) otherwise. MCTS should discover that picking fewer
        features can be better than picking more.
        """
        nck = NChooseK(features=[0, 1, 2, 3, 4, 5], min_count=1, max_count=4)
        gs = Groups(groups=[nck])

        def reward_fn(feats, _cats):
            if set(feats) == {2, 5}:
                return 80.0
            return float(len(feats))

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=42)
        feats, _cats, val = mcts.run(n_iterations=300)
        assert val == 80.0
        assert set(feats) == {2, 5}

    def test_large_categorical_space(self):
        """Three categoricals with 4 values each (64 combinations).

        Only one combination is optimal.
        """
        cat1 = Categorical(dim=0, values=[0.0, 1.0, 2.0, 3.0])
        cat2 = Categorical(dim=1, values=[0.0, 1.0, 2.0, 3.0])
        cat3 = Categorical(dim=2, values=[0.0, 1.0, 2.0, 3.0])
        gs = Groups(groups=[cat1, cat2, cat3])

        target = {0: 3.0, 1: 1.0, 2: 2.0}

        def reward_fn(_feats, cats):
            if cats == target:
                return 100.0
            # Partial credit: +10 per matching dim
            return sum(10.0 for d in target if cats.get(d) == target[d])

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=99)
        feats, cats, val = mcts.run(n_iterations=200)
        assert val == 100.0
        assert cats == target
        assert feats == ()

    def test_multiple_nchoosek_groups(self):
        """Two independent NChooseK groups, each with its own optimum."""
        nck1 = NChooseK(features=[0, 1, 2], min_count=1, max_count=1)
        nck2 = NChooseK(features=[10, 11, 12], min_count=1, max_count=1)
        gs = Groups(groups=[nck1, nck2])

        def reward_fn(feats, _cats):
            # Optimum: feature 2 from group 1, feature 11 from group 2
            if set(feats) == {2, 11}:
                return 100.0
            score = 0.0
            if 2 in feats:
                score += 40.0
            if 11 in feats:
                score += 40.0
            return score

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=0)
        feats, _cats, val = mcts.run(n_iterations=200)
        assert val == 100.0
        assert set(feats) == {2, 11}

    def test_convergence_with_noisy_reward(self):
        """Reward has a noise floor but the optimum still dominates.

        Verifies MCTS handles reward variance and still identifies the best.
        """
        import random as stdlib_random

        rng = stdlib_random.Random(12345)
        nck = NChooseK(features=[0, 1, 2], min_count=1, max_count=1)
        gs = Groups(groups=[nck])

        def reward_fn(feats, _cats):
            noise = rng.uniform(-1.0, 1.0)
            if 2 in feats:
                return 50.0 + noise
            if 1 in feats:
                return 20.0 + noise
            return 5.0 + noise

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=42)
        feats, _cats, val = mcts.run(n_iterations=150)
        # The best encountered value should come from feature 2
        assert 2 in feats
        assert val >= 49.0  # 50 - max noise

    def test_mcts_outperforms_random_rollouts(self):
        """MCTS with UCT/backprop should achieve higher average best
        reward than pure random rollouts on a structured problem.

        Problem: pick 1-4 features from a pool of 10.  The search space
        includes all subsets of size 1 through 4, which is
        C(10,1)+C(10,2)+C(10,3)+C(10,4) = 10+45+120+210 = 385
        combinations.  The optimal subset {2, 7} has size 2 (not the max),
        so MCTS must learn both *which* features and *how many* to select.
        Partial credit steers MCTS toward the right features.
        """
        nck = NChooseK(features=list(range(10)), min_count=1, max_count=4)
        gs = Groups(groups=[nck])
        optimal = {2, 7}
        budget = 60

        def reward_fn(feats, _cats):
            feat_set = set(feats)
            if feat_set == optimal:
                return 100.0
            # Partial credit: reward correct features, penalise extras
            overlap = len(feat_set & optimal)
            extras = len(feat_set - optimal)
            return float(overlap * 30 - extras * 10)

        # --- random baseline: repeated rollouts from root, keep best ---
        def random_search(seed: int) -> float:
            mcts_tmp = MCTS(groups=gs, reward_fn=lambda f, c: 0.0, seed=seed)
            rng = stdlib_random.Random(seed)
            best = float("-inf")
            for _ in range(budget):
                mcts_tmp.rng = rng
                feats, cats, _traj = mcts_tmp._rollout(mcts_tmp.root)
                val = reward_fn(feats, cats)
                if val > best:
                    best = val
            return best

        n_trials = 20
        mcts_best_vals = []
        random_best_vals = []

        for trial_seed in range(n_trials):
            # MCTS run
            mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=trial_seed)
            _feats, _cats, mcts_val = mcts.run(n_iterations=budget)
            mcts_best_vals.append(mcts_val)

            # Random baseline with separate seed space
            random_val = random_search(seed=trial_seed + 1000)
            random_best_vals.append(random_val)

        mcts_mean = sum(mcts_best_vals) / n_trials
        random_mean = sum(random_best_vals) / n_trials

        # MCTS should achieve a meaningfully higher average best reward
        assert mcts_mean > random_mean, (
            f"MCTS mean best {mcts_mean:.1f} should beat "
            f"random mean best {random_mean:.1f}"
        )

    def test_mcts_outperforms_random_mixed_problem(self):
        """Mixed NChooseK + Categorical problem where MCTS should beat random.

        2 NChooseK groups + 2 categoricals = large combinatorial space.
        MCTS should exploit partial-credit structure to converge faster.
        """
        nck1 = NChooseK(features=[0, 1, 2, 3], min_count=1, max_count=2)
        nck2 = NChooseK(features=[10, 11, 12, 13], min_count=1, max_count=2)
        cat1 = Categorical(dim=20, values=[0.0, 1.0, 2.0, 3.0])
        cat2 = Categorical(dim=21, values=[0.0, 1.0, 2.0])
        gs = Groups(groups=[nck1, nck2, cat1, cat2])

        target_feats = {1, 3, 11}
        target_cats = {20: 2.0, 21: 1.0}
        budget = 60

        def reward_fn(feats, cats):
            feat_set = set(feats)
            # Partial credit: 10 per correct feature, 15 per correct categorical
            score = sum(10.0 for f in target_feats if f in feat_set)
            score += sum(15.0 for d, v in target_cats.items() if cats.get(d) == v)
            # Penalty for wrong features
            score -= 5.0 * len(feat_set - target_feats)
            # Bonus for exact match
            if feat_set == target_feats and cats == target_cats:
                score = 100.0
            return score

        n_trials = 15
        mcts_best_vals = []
        random_best_vals = []

        for trial_seed in range(n_trials):
            # MCTS
            mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=trial_seed)
            _f, _c, mcts_val = mcts.run(n_iterations=budget)
            mcts_best_vals.append(mcts_val)

            # Random baseline
            mcts_tmp = MCTS(
                groups=gs, reward_fn=lambda f, c: 0.0, seed=trial_seed + 500
            )
            rng = stdlib_random.Random(trial_seed + 500)
            best_rand = float("-inf")
            for _ in range(budget):
                mcts_tmp.rng = rng
                feats, cats, _traj = mcts_tmp._rollout(mcts_tmp.root)
                val = reward_fn(feats, cats)
                if val > best_rand:
                    best_rand = val
            random_best_vals.append(best_rand)

        mcts_mean = sum(mcts_best_vals) / n_trials
        random_mean = sum(random_best_vals) / n_trials

        assert mcts_mean > random_mean, (
            f"MCTS mean best {mcts_mean:.1f} should beat "
            f"random mean best {random_mean:.1f}"
        )


# =============================================================================
# optimize_acqf_mcts tests
# =============================================================================


def _make_mock_optimize_acqf(d: int, q: int = 1):
    """Create a mock for botorch.optim.optimize_acqf.

    Returns (mock_fn, call_log) where call_log is a list that collects
    the keyword arguments of every call.  The mock returns random
    candidates and uses the sum of non-fixed dimensions as the acq value
    so different fixed-feature configs yield different rewards.
    """
    call_log: list[dict] = []

    def mock_fn(**kwargs):
        call_log.append(kwargs)
        fixed = kwargs.get("fixed_features") or {}
        # Build a candidate tensor respecting fixed features
        cand = torch.rand(q, d)
        for dim, val in fixed.items():
            cand[:, dim] = val
        # Reward = sum of candidate values (so fixing dims to 0 lowers it)
        acq_val = cand.sum()
        return cand, acq_val

    return mock_fn, call_log


class TestOptimizeAcqfMcts:
    """Tests for the top-level optimize_acqf_mcts function."""

    # ---- output shape and type ----

    def test_returns_correct_shape_nchoosek(self, monkeypatch):
        d = 5
        mock_fn, _ = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        candidates, acq_val = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1, 2], 1, 2)],
            num_iterations=10,
            seed=0,
        )
        assert candidates.shape == (1, d)
        assert isinstance(acq_val, float)

    def test_returns_correct_shape_q_greater_than_one(self, monkeypatch):
        d, q = 4, 3
        mock_fn, _ = _make_mock_optimize_acqf(d, q=q)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        candidates, acq_val = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1, 2], 1, 2)],
            q=q,
            num_iterations=10,
            seed=0,
        )
        assert candidates.shape == (q, d)

    # ---- zero iterations fallback ----

    def test_zero_iterations_returns_zeros(self, monkeypatch):
        d = 4
        call_log: list[dict] = []

        def mock_fn(**kwargs):
            call_log.append(kwargs)
            return torch.zeros(1, d), torch.tensor(0.0)

        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        candidates, acq_val = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1, 2], 1, 2)],
            num_iterations=0,
            seed=0,
        )
        # No iterations => optimize_acqf never called
        assert len(call_log) == 0
        assert candidates.shape == (1, d)
        assert torch.all(candidates == 0)
        assert acq_val == float("-inf")

    # ---- NChooseK: inactive features fixed to zero ----

    def test_inactive_features_fixed_to_zero(self, monkeypatch):
        """Verify that features NOT selected by MCTS are fixed to 0.0."""
        d = 5
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        nchoosek_features = [0, 1, 2, 3]

        optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[(nchoosek_features, 1, 2)],
            num_iterations=15,
            seed=0,
        )

        assert len(call_log) > 0
        for call_kwargs in call_log:
            fixed = call_kwargs["fixed_features"]
            # Every nchoosek feature is either fixed to 0 (inactive) or
            # absent from fixed (active and free to optimize)
            for f in nchoosek_features:
                if f in fixed:
                    assert fixed[f] == 0.0

    def test_active_features_not_fixed(self, monkeypatch):
        """At least one call should leave some features free (not in fixed)."""
        d = 4
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        nchoosek_features = [0, 1, 2, 3]

        optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[(nchoosek_features, 1, 3)],
            num_iterations=20,
            seed=0,
        )

        # In every call, at least min_count features should be free
        for call_kwargs in call_log:
            fixed = call_kwargs["fixed_features"]
            n_fixed_nck = sum(1 for f in nchoosek_features if f in fixed)
            n_free = len(nchoosek_features) - n_fixed_nck
            assert n_free >= 1  # min_count = 1

    # ---- Categorical: dims fixed to selected value ----

    def test_categorical_dims_fixed_to_allowed_value(self, monkeypatch):
        d = 5
        cat_dims = {3: [0.0, 1.0, 2.0], 4: [10.0, 20.0]}
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            cat_dims=cat_dims,
            num_iterations=15,
            seed=0,
        )

        assert len(call_log) > 0
        for call_kwargs in call_log:
            fixed = call_kwargs["fixed_features"]
            # Every categorical dim should be fixed to one of its allowed values
            for dim, allowed in cat_dims.items():
                assert dim in fixed, f"Categorical dim {dim} not in fixed_features"
                assert fixed[dim] in allowed, (
                    f"Categorical dim {dim} fixed to {fixed[dim]}, "
                    f"expected one of {allowed}"
                )

    # ---- Mixed NChooseK + Categorical ----

    def test_mixed_nchoosek_and_categorical(self, monkeypatch):
        d = 6
        nchoosek_features = [0, 1, 2]
        cat_dims = {4: [0.0, 1.0], 5: [10.0, 20.0, 30.0]}
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        candidates, acq_val = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[(nchoosek_features, 1, 2)],
            cat_dims=cat_dims,
            num_iterations=15,
            seed=0,
        )

        assert candidates.shape == (1, d)
        assert len(call_log) > 0
        for call_kwargs in call_log:
            fixed = call_kwargs["fixed_features"]
            # Categorical dims are always fixed
            assert 4 in fixed
            assert 5 in fixed
            # NChooseK features are either active (free) or fixed to 0
            for f in nchoosek_features:
                if f in fixed:
                    assert fixed[f] == 0.0

    # ---- User-provided fixed_features are forwarded ----

    def test_user_fixed_features_forwarded(self, monkeypatch):
        d = 5
        user_fixed = {4: 99.0}
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1, 2], 1, 2)],
            fixed_features=user_fixed,
            num_iterations=10,
            seed=0,
        )

        for call_kwargs in call_log:
            fixed = call_kwargs["fixed_features"]
            # User's fixed feature should always be present
            assert fixed[4] == 99.0

    # ---- Constraints are forwarded ----

    def test_constraints_forwarded(self, monkeypatch):
        d = 4
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        ineq = [(torch.tensor([0]), torch.tensor([1.0]), 0.5)]
        eq = [(torch.tensor([1]), torch.tensor([1.0]), 1.0)]

        optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1, 2], 1, 2)],
            inequality_constraints=ineq,
            equality_constraints=eq,
            num_iterations=5,
            seed=0,
        )

        assert len(call_log) > 0
        for call_kwargs in call_log:
            assert call_kwargs["inequality_constraints"] is ineq
            assert call_kwargs["equality_constraints"] is eq

    # ---- BoTorch params forwarded ----

    def test_botorch_params_forwarded(self, monkeypatch):
        d = 4
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        optimize_acqf_mcts(
            acq_function="sentinel_acqf",
            bounds=bounds,
            nchooseks=[([0, 1], 1, 1)],
            q=2,
            raw_samples=512,
            num_restarts=8,
            num_iterations=5,
            seed=0,
        )

        for call_kwargs in call_log:
            assert call_kwargs["acq_function"] == "sentinel_acqf"
            assert call_kwargs["q"] == 2
            assert call_kwargs["raw_samples"] == 512
            assert call_kwargs["num_restarts"] == 8

    # ---- Bounds forwarded correctly ----

    def test_bounds_forwarded(self, monkeypatch):
        d = 3
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.tensor([[0.0, -1.0, 0.0], [1.0, 1.0, 5.0]])

        optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1], 1, 1)],
            num_iterations=5,
            seed=0,
        )

        for call_kwargs in call_log:
            assert torch.equal(call_kwargs["bounds"], bounds)

    # ---- Multiple NChooseK groups ----

    def test_multiple_nchoosek_groups(self, monkeypatch):
        d = 6
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        nchooseks = [
            ([0, 1, 2], 1, 2),  # group 1
            ([3, 4, 5], 1, 1),  # group 2
        ]

        candidates, acq_val = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=nchooseks,
            num_iterations=15,
            seed=0,
        )

        assert candidates.shape == (1, d)
        all_nck_features = {0, 1, 2, 3, 4, 5}
        for call_kwargs in call_log:
            fixed = call_kwargs["fixed_features"]
            # Each NChooseK feature is either fixed to 0 or free
            for f in all_nck_features:
                if f in fixed:
                    assert fixed[f] == 0.0
            # Group 2 requires exactly 1 free feature from [3,4,5]
            n_free_g2 = sum(1 for f in [3, 4, 5] if f not in fixed)
            assert n_free_g2 == 1

    # ---- No constraints at all (only bounds) ----

    def test_no_nchoosek_no_categoricals(self, monkeypatch):
        """With neither NChooseK nor categoricals, optimize_acqf is still called."""
        d = 3
        mock_fn, call_log = _make_mock_optimize_acqf(d)
        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        candidates, acq_val = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            num_iterations=5,
            seed=0,
        )

        assert candidates.shape == (1, d)
        # With no groups, every MCTS iteration is immediately terminal
        # so optimize_acqf should be called on every iteration
        assert len(call_log) > 0
        for call_kwargs in call_log:
            # No features should be fixed (no NChooseK, no categoricals)
            fixed = call_kwargs["fixed_features"]
            assert fixed == {}

    # ---- Best candidate is from the call with highest acq value ----

    def test_best_candidate_tracks_highest_acq_value(self, monkeypatch):
        """The returned candidates should come from the optimize_acqf call
        with the highest acquisition value."""
        d = 3
        call_count = 0
        best_cand = torch.tensor([[0.1, 0.2, 0.3]])

        def mock_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            # Return a high value on the 3rd call, low otherwise
            if call_count == 3:
                return best_cand, torch.tensor(999.0)
            return torch.rand(1, d), torch.tensor(1.0)

        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", mock_fn)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])

        candidates, acq_val = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1, 2], 1, 2)],
            num_iterations=10,
            seed=0,
        )

        assert acq_val == 999.0
        assert torch.equal(candidates, best_cand)

    # ---- dtype preserved ----

    def test_dtype_preserved_on_zero_iterations(self, monkeypatch):
        """When num_iterations=0, the fallback zeros should match bounds dtype."""
        d = 3

        def noop(**kwargs):
            return torch.zeros(1, d), torch.tensor(0.0)

        monkeypatch.setattr(optimize_mcts_mod, "optimize_acqf", noop)
        bounds = torch.stack(
            [torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64)]
        )

        candidates, _ = optimize_acqf_mcts(
            acq_function=None,
            bounds=bounds,
            nchooseks=[([0, 1], 1, 1)],
            num_iterations=0,
        )
        assert candidates.dtype == torch.float64
