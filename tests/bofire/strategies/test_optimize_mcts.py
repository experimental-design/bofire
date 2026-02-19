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
        assert mcts.c_uct == 1.0
        assert mcts.k_rave == 300.0
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
            k_rave=100.0,
            p_stop_rollout=0.5,
            pw_k0=3.0,
            pw_alpha=0.7,
            seed=42,
        )
        assert mcts.c_uct == 2.0
        assert mcts.k_rave == 100.0
        assert mcts.p_stop_rollout == 0.5
        assert mcts.pw_k0 == 3.0
        assert mcts.pw_alpha == 0.7

    # ---- _compute_group_offsets ----

    def test_group_offsets_single_group(self):
        gs = self._nck_only_groups()  # 3 options
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        assert mcts.global_offsets == [0]

    def test_group_offsets_mixed(self):
        gs = self._mixed_groups()  # NChooseK(3 options) + Categorical(2 options)
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        assert mcts.global_offsets == [0, 3]

    def test_group_offsets_multiple_nchoosek(self):
        nck1 = NChooseK(features=[0, 1], min_count=1, max_count=1)  # 2 options
        nck2 = NChooseK(features=[5, 6, 7], min_count=1, max_count=2)  # 3 options
        cat = Categorical(dim=3, values=[0.0, 1.0, 2.0])  # 3 options
        gs = Groups(groups=[nck1, nck2, cat])
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        assert mcts.global_offsets == [0, 2, 5]

    def test_group_offsets_empty(self):
        gs = Groups(groups=[])
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        assert mcts.global_offsets == []

    # ---- _global_action_id ----

    def test_global_action_id(self):
        gs = self._mixed_groups()  # offsets [0, 3]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        assert mcts._global_action_id(0, 0) == 0
        assert mcts._global_action_id(0, 2) == 2
        assert mcts._global_action_id(1, 0) == 3
        assert mcts._global_action_id(1, 1) == 4

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
        feats, cats = mcts._rollout(mcts.root)
        # Should return a valid selection with at least min_count=1 features
        assert len(feats) >= 1
        assert all(f in [0, 1, 2] for f in feats)
        assert cats == {}

    def test_rollout_categorical_selects_one(self):
        gs = self._cat_only_groups()  # dim=0, values=[10,20,30]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=7)
        feats, cats = mcts._rollout(mcts.root)
        assert feats == ()
        assert 0 in cats
        assert cats[0] in [10.0, 20.0, 30.0]

    def test_rollout_mixed(self):
        gs = self._mixed_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward(), seed=99)
        feats, cats = mcts._rollout(mcts.root)
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
        mcts._backpropagate(path, reward=3.0, selected_features=(0,), cat_selections={})
        assert root.n_visits == 1
        assert root.w_total == 3.0
        assert child.n_visits == 1
        assert child.w_total == 3.0

    def test_backpropagate_accumulates(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        root = mcts.root
        path = [root]
        mcts._backpropagate(path, reward=2.0, selected_features=(0,), cat_selections={})
        mcts._backpropagate(path, reward=5.0, selected_features=(1,), cat_selections={})
        assert root.n_visits == 2
        assert root.w_total == pytest.approx(7.0)

    def test_backpropagate_updates_rave_nchoosek(self):
        gs = self._nck_only_groups()  # features=[0,1,2], offsets=[0]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        path = [mcts.root]
        # Selected feature 0 (local index 0) and feature 2 (local index 2)
        mcts._backpropagate(
            path, reward=4.0, selected_features=(0, 2), cat_selections={}
        )
        # RAVE should be updated for global IDs 0 and 2
        assert mcts.rave_stats[0] == (1, 4.0)
        assert mcts.rave_stats[2] == (1, 4.0)
        # Feature 1 (global ID 1) not selected
        assert 1 not in mcts.rave_stats

    def test_backpropagate_updates_rave_categorical(self):
        gs = self._cat_only_groups()  # dim=0, values=[10,20,30], offset=[0]
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        path = [mcts.root]
        # Selected value 20.0 which is index 1
        mcts._backpropagate(
            path, reward=6.0, selected_features=(), cat_selections={0: 20.0}
        )
        assert mcts.rave_stats[1] == (1, 6.0)
        # Indices 0 and 2 not selected
        assert 0 not in mcts.rave_stats
        assert 2 not in mcts.rave_stats

    def test_backpropagate_rave_accumulates(self):
        gs = self._nck_only_groups()
        mcts = MCTS(groups=gs, reward_fn=self._constant_reward())
        path = [mcts.root]
        mcts._backpropagate(path, reward=2.0, selected_features=(0,), cat_selections={})
        mcts._backpropagate(path, reward=8.0, selected_features=(0,), cat_selections={})
        assert mcts.rave_stats[0] == (2, 10.0)

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
            mcts._backpropagate(
                path, reward=1.0, selected_features=(), cat_selections={}
            )
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
        assert mcts.root.n_visits == n_iter

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

        mcts = MCTS(groups=gs, reward_fn=reward_fn, seed=0)
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
        """MCTS with UCT/RAVE/backprop should achieve higher average best
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
                feats, cats = mcts_tmp._rollout(mcts_tmp.root)
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
                feats, cats = mcts_tmp._rollout(mcts_tmp.root)
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
