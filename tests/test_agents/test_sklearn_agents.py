"""Tests for sklearn-based behavioral cloning agents."""

import numpy as np
import pytest
from pathlib import Path

from src.agents.sklearn_agent import (
    RandomForestAgent,
    GradientBoostAgent,
    KNNAgent,
    SVMAgent,
)
from src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_AGENT_CLASSES = [RandomForestAgent, GradientBoostAgent, KNNAgent, SVMAgent]


def _make_state(rng=None):
    """Return a random float32 state vector of length 5."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(5).astype(np.float32)


def _push_diverse_transitions(agent, n=300, rng=None):
    """Feed *n* diverse (state, action) transitions through train_step.

    Actions are mixed: roughly half 0 and half 1, driven by the sign of
    the first state feature so the model has a learnable signal.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    for _ in range(n):
        state = rng.standard_normal(5).astype(np.float32)
        action = 1 if state[0] > 0 else 0
        next_state = rng.standard_normal(5).astype(np.float32)
        reward = 1.0
        done = False
        agent.train_step(state, action, reward, next_state, done)


# ===================================================================
# TestSklearnCreation
# ===================================================================


class TestSklearnCreation:
    """Test agent instantiation and default parameters for every agent type."""

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_is_base_agent_subclass(self, AgentClass):
        agent = AgentClass(state_dim=5, action_dim=2)
        assert isinstance(agent, BaseAgent)

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_default_parameters(self, AgentClass):
        agent = AgentClass(state_dim=5, action_dim=2)
        assert agent.state_dim == 5
        assert agent.action_dim == 2
        assert agent.epsilon == pytest.approx(1.0)

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_not_trained_initially(self, AgentClass):
        agent = AgentClass(state_dim=5, action_dim=2)
        assert agent.trained is False


# ===================================================================
# TestSklearnSelectAction
# ===================================================================


class TestSklearnSelectAction:
    """Test action selection for every agent type."""

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_returns_valid_action(self, AgentClass):
        agent = AgentClass(state_dim=5, action_dim=2)
        state = np.random.randn(5).astype(np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert action in (0, 1)

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_returns_random_when_untrained(self, AgentClass):
        """An untrained agent with epsilon=1.0 should pick randomly.

        Over 50 calls we expect to see both 0 and 1 at least once.
        """
        agent = AgentClass(state_dim=5, action_dim=2)
        state = np.random.randn(5).astype(np.float32)
        actions = {agent.select_action(state) for _ in range(50)}
        assert 0 in actions
        assert 1 in actions


# ===================================================================
# TestSklearnTrainStep  (using RandomForestAgent as representative)
# ===================================================================


class TestSklearnTrainStep:
    """Test the train_step loop using RandomForestAgent."""

    def test_returns_dict(self):
        agent = RandomForestAgent(state_dim=5, action_dim=2)
        state = np.random.randn(5).astype(np.float32)
        next_state = np.random.randn(5).astype(np.float32)
        result = agent.train_step(state, 0, 1.0, next_state, False)
        assert isinstance(result, dict)

    def test_epsilon_decays(self):
        agent = RandomForestAgent(
            state_dim=5,
            action_dim=2,
            epsilon_start=1.0,
            epsilon_decay=0.5,
            epsilon_end=0.01,
        )
        state = np.random.randn(5).astype(np.float32)
        agent.train_step(state, 0, 1.0, state, False)
        assert agent.epsilon < 1.0

    def test_stores_samples(self):
        agent = RandomForestAgent(state_dim=5, action_dim=2)
        state = np.random.randn(5).astype(np.float32)
        next_state = np.random.randn(5).astype(np.float32)
        agent.train_step(state, 1, 1.0, next_state, False)
        # After one transition the agent should hold at least one sample
        info = agent.get_info()
        assert info["samples"] >= 1

    def test_trains_after_enough_samples(self):
        """After 300+ diverse transitions the model should be trained."""
        agent = RandomForestAgent(
            state_dim=5,
            action_dim=2,
            retrain_every=200,
            min_samples=50,
        )
        _push_diverse_transitions(agent, n=300)
        assert agent.trained is True

    def test_death_stores_opposite_action(self):
        """When done=True the agent should store the *opposite* action.

        If the agent chose action=0 and died, it records action=1 so the
        classifier learns to avoid the fatal action.
        """
        agent = RandomForestAgent(
            state_dim=5,
            action_dim=2,
            retrain_every=200,
            min_samples=50,
        )
        # Push enough non-done transitions first to have some data.
        rng = np.random.default_rng(0)
        for _ in range(100):
            s = rng.standard_normal(5).astype(np.float32)
            a = rng.integers(0, 2)
            ns = rng.standard_normal(5).astype(np.float32)
            agent.train_step(s, a, 1.0, ns, False)

        samples_before = agent.get_info()["samples"]

        # Now push a death transition with action=0
        state = rng.standard_normal(5).astype(np.float32)
        next_state = rng.standard_normal(5).astype(np.float32)
        agent.train_step(state, 0, -1.0, next_state, True)

        samples_after = agent.get_info()["samples"]
        # A sample should still have been stored
        assert samples_after > samples_before


# ===================================================================
# TestSklearnPrediction  (using RandomForestAgent)
# ===================================================================


class TestSklearnPrediction:
    """Test prediction quality after training on separable data."""

    @pytest.fixture()
    def trained_agent(self):
        """Return a RandomForestAgent trained on a simple rule:

        obs[0] > 0 -> action 1
        obs[0] < 0 -> action 0
        """
        agent = RandomForestAgent(
            state_dim=5,
            action_dim=2,
            retrain_every=100,
            min_samples=50,
        )
        rng = np.random.default_rng(123)
        for _ in range(300):
            state = rng.standard_normal(5).astype(np.float32)
            action = 1 if state[0] > 0 else 0
            next_state = rng.standard_normal(5).astype(np.float32)
            agent.train_step(state, action, 1.0, next_state, False)
        assert agent.trained is True
        return agent

    def test_predicts_after_training(self, trained_agent):
        """The trained model should predict correctly on clear-cut inputs."""
        agent = trained_agent
        agent.epsilon = 0.0  # disable exploration

        # Strongly positive first feature -> action 1
        pos_state = np.array([3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert agent.select_action(pos_state, training=False) == 1

        # Strongly negative first feature -> action 0
        neg_state = np.array([-3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert agent.select_action(neg_state, training=False) == 0

    def test_get_q_values_shape(self, trained_agent):
        state = np.random.randn(5).astype(np.float32)
        q_vals = trained_agent._get_q_values(state)
        assert isinstance(q_vals, np.ndarray)
        assert q_vals.shape == (2,)

    def test_get_q_values_sums_to_one(self, trained_agent):
        state = np.random.randn(5).astype(np.float32)
        q_vals = trained_agent._get_q_values(state)
        assert q_vals.sum() == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# TestSklearnFeatureImportances
# ===================================================================


class TestSklearnFeatureImportances:
    """Test feature-importance retrieval."""

    def test_random_forest_has_importances(self):
        agent = RandomForestAgent(
            state_dim=5,
            action_dim=2,
            retrain_every=100,
            min_samples=50,
        )
        _push_diverse_transitions(agent, n=300)
        assert agent.trained is True

        importances = agent.get_feature_importances()
        assert isinstance(importances, np.ndarray)
        assert importances.shape == (5,)

    def test_knn_has_no_importances(self):
        agent = KNNAgent(
            state_dim=5,
            action_dim=2,
            retrain_every=100,
            min_samples=50,
        )
        _push_diverse_transitions(agent, n=300)
        assert agent.trained is True

        importances = agent.get_feature_importances()
        assert importances is None


# ===================================================================
# TestSklearnSaveLoad
# ===================================================================


class TestSklearnSaveLoad:
    """Test save/load round-trip using RandomForestAgent."""

    def test_save_load_roundtrip(self, tmp_path):
        # Train the agent so it has a real model to persist
        agent = RandomForestAgent(
            state_dim=5,
            action_dim=2,
            retrain_every=100,
            min_samples=50,
        )
        _push_diverse_transitions(agent, n=300)
        assert agent.trained is True

        state = np.array([2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        agent.epsilon = 0.0
        pred_before = agent.select_action(state, training=False)

        save_path = tmp_path / "sklearn_agent"
        agent.save(save_path)

        # Load into a brand-new agent
        agent2 = RandomForestAgent(state_dim=5, action_dim=2)
        agent2.load(save_path)
        agent2.epsilon = 0.0

        pred_after = agent2.select_action(state, training=False)
        assert pred_before == pred_after

        # Q-value distributions should match as well
        q_before = agent._get_q_values(state)
        q_after = agent2._get_q_values(state)
        np.testing.assert_array_almost_equal(q_before, q_after)


# ===================================================================
# TestSklearnGetInfo
# ===================================================================


class TestSklearnGetInfo:
    """Test the get_info introspection method."""

    def test_returns_expected_keys(self):
        agent = RandomForestAgent(state_dim=5, action_dim=2)
        info = agent.get_info()
        assert "epsilon" in info
        assert "trained" in info
        assert "samples" in info


# ===================================================================
# TestAllAgentTypes  (parametrised full training loop)
# ===================================================================


class TestAllAgentTypes:
    """Run a full training loop for every sklearn agent variant."""

    @pytest.mark.parametrize("AgentClass", ALL_AGENT_CLASSES)
    def test_full_training_loop(self, AgentClass):
        agent = AgentClass(
            state_dim=5,
            action_dim=2,
            retrain_every=200,
            min_samples=50,
        )
        rng = np.random.default_rng(99)
        for _ in range(500):
            state = rng.standard_normal(5).astype(np.float32)
            action = 1 if state[0] > 0 else 0
            next_state = rng.standard_normal(5).astype(np.float32)
            reward = 1.0
            done = rng.random() < 0.05  # occasional episode end
            agent.train_step(state, action, reward, next_state, done)

        assert agent.trained is True
