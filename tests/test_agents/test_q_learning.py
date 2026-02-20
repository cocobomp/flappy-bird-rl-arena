"""Tests for Q-Learning agent."""

import json
import pickle
import numpy as np
import pytest
from pathlib import Path

from src.agents.q_learning import QLearningAgent
from src.agents.base_agent import BaseAgent


class TestQLearningCreation:
    """Test agent creation and default parameters."""

    def test_is_base_agent_subclass(self):
        agent = QLearningAgent(state_dim=4, action_dim=2)
        assert isinstance(agent, BaseAgent)

    def test_default_parameters(self):
        agent = QLearningAgent(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.n_bins == 10
        assert agent.lr == 0.1
        assert agent.gamma == 0.99
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay == 0.995

    def test_custom_parameters(self):
        agent = QLearningAgent(
            state_dim=8, action_dim=3, n_bins=20,
            lr=0.5, gamma=0.9, epsilon_start=0.5,
            epsilon_end=0.05, epsilon_decay=0.99,
        )
        assert agent.state_dim == 8
        assert agent.action_dim == 3
        assert agent.n_bins == 20
        assert agent.lr == 0.5
        assert agent.gamma == 0.9
        assert agent.epsilon == 0.5
        assert agent.epsilon_end == 0.05
        assert agent.epsilon_decay == 0.99

    def test_q_table_starts_empty(self):
        agent = QLearningAgent(state_dim=4, action_dim=2)
        assert len(agent.q_table) == 0


class TestDiscretize:
    """Test state discretization."""

    def test_discretize_returns_tuple(self):
        agent = QLearningAgent(state_dim=3, action_dim=2, n_bins=5)
        state = np.array([0.0, 0.5, -0.5])
        result = agent._discretize(state)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_discretize_clips_to_range(self):
        agent = QLearningAgent(state_dim=2, action_dim=2, n_bins=5)
        # Values outside [-1, 1] should be clipped
        state_big = np.array([10.0, -10.0])
        state_edge = np.array([1.0, -1.0])
        result_big = agent._discretize(state_big)
        result_edge = agent._discretize(state_edge)
        assert result_big == result_edge

    def test_discretize_different_states_differ(self):
        agent = QLearningAgent(state_dim=2, action_dim=2, n_bins=10)
        s1 = np.array([0.0, 0.0])
        s2 = np.array([0.9, -0.9])
        assert agent._discretize(s1) != agent._discretize(s2)


class TestSelectAction:
    """Test action selection (epsilon-greedy)."""

    def test_returns_valid_action(self):
        agent = QLearningAgent(state_dim=4, action_dim=3)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        for _ in range(50):
            action = agent.select_action(state)
            assert 0 <= action < 3

    def test_greedy_with_epsilon_zero(self):
        agent = QLearningAgent(state_dim=2, action_dim=3, n_bins=5)
        agent.epsilon = 0.0
        state = np.array([0.5, 0.5])
        key = agent._discretize(state)
        # Set Q-values so action 2 is clearly best
        agent.q_table[key] = np.array([0.0, 0.0, 10.0])

        # With epsilon=0, should always pick action 2
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        assert all(a == 2 for a in actions)

    def test_greedy_when_not_training(self):
        agent = QLearningAgent(state_dim=2, action_dim=3, n_bins=5)
        agent.epsilon = 1.0  # Would always explore if training
        state = np.array([0.5, 0.5])
        key = agent._discretize(state)
        agent.q_table[key] = np.array([0.0, 0.0, 10.0])

        # With training=False, should always pick greedy (action 2)
        actions = [agent.select_action(state, training=False) for _ in range(100)]
        assert all(a == 2 for a in actions)

    def test_explores_with_high_epsilon(self):
        agent = QLearningAgent(state_dim=2, action_dim=3, n_bins=5)
        agent.epsilon = 1.0
        state = np.array([0.5, 0.5])
        key = agent._discretize(state)
        agent.q_table[key] = np.array([0.0, 0.0, 10.0])

        # With epsilon=1.0 and training, should see some variety
        actions = set(agent.select_action(state, training=True) for _ in range(200))
        assert len(actions) > 1


class TestTrainStep:
    """Test Q-learning update rule."""

    def test_returns_dict(self):
        agent = QLearningAgent(state_dim=2, action_dim=2)
        state = np.array([0.0, 0.0])
        result = agent.train_step(state, 0, 1.0, np.array([0.1, 0.1]), False)
        assert isinstance(result, dict)

    def test_q_update_simple(self):
        """With lr=1.0 and gamma=0.0, Q(s,a) should become exactly r."""
        agent = QLearningAgent(
            state_dim=2, action_dim=2,
            n_bins=5, lr=1.0, gamma=0.0,
        )
        state = np.array([0.5, 0.5])
        next_state = np.array([0.6, 0.6])
        reward = 5.0

        agent.train_step(state, 0, reward, next_state, False)

        key = agent._discretize(state)
        assert agent.q_table[key][0] == pytest.approx(5.0)
        # Action 1 should still be 0
        assert agent.q_table[key][1] == pytest.approx(0.0)

    def test_q_update_with_gamma(self):
        """With lr=1.0 and gamma=0.5, Q(s,a) = r + gamma * max(Q(s'))."""
        agent = QLearningAgent(
            state_dim=2, action_dim=2,
            n_bins=5, lr=1.0, gamma=0.5,
        )
        next_state = np.array([0.6, 0.6])
        next_key = agent._discretize(next_state)
        # Pre-load next state Q-values
        agent.q_table[next_key] = np.array([4.0, 2.0])

        state = np.array([0.5, 0.5])
        agent.train_step(state, 0, 1.0, next_state, False)

        key = agent._discretize(state)
        # Q = 0 + 1.0*(1.0 + 0.5*4.0 - 0) = 3.0
        assert agent.q_table[key][0] == pytest.approx(3.0)

    def test_q_update_done_ignores_next(self):
        """When done=True, next state max Q should not contribute."""
        agent = QLearningAgent(
            state_dim=2, action_dim=2,
            n_bins=5, lr=1.0, gamma=0.99,
        )
        next_state = np.array([0.6, 0.6])
        next_key = agent._discretize(next_state)
        agent.q_table[next_key] = np.array([100.0, 100.0])

        state = np.array([0.5, 0.5])
        agent.train_step(state, 0, 2.0, next_state, True)

        key = agent._discretize(state)
        # done=True: Q = 0 + 1.0*(2.0 + 0 - 0) = 2.0
        assert agent.q_table[key][0] == pytest.approx(2.0)

    def test_epsilon_decays(self):
        agent = QLearningAgent(
            state_dim=2, action_dim=2,
            epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.01,
        )
        state = np.array([0.0, 0.0])
        agent.train_step(state, 0, 0.0, state, False)
        assert agent.epsilon == pytest.approx(0.5)
        agent.train_step(state, 0, 0.0, state, False)
        assert agent.epsilon == pytest.approx(0.25)

    def test_epsilon_does_not_go_below_min(self):
        agent = QLearningAgent(
            state_dim=2, action_dim=2,
            epsilon_start=0.02, epsilon_decay=0.1, epsilon_end=0.01,
        )
        state = np.array([0.0, 0.0])
        agent.train_step(state, 0, 0.0, state, False)
        assert agent.epsilon >= 0.01


class TestSaveLoad:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        agent = QLearningAgent(state_dim=2, action_dim=3, n_bins=5)
        state = np.array([0.5, 0.5])

        # Train a bit to populate Q-table
        for _ in range(10):
            action = agent.select_action(state)
            agent.train_step(state, action, 1.0, state, False)

        original_epsilon = agent.epsilon
        original_q_table_size = len(agent.q_table)

        save_path = tmp_path / "q_agent"
        agent.save(save_path)

        # Create a fresh agent and load
        agent2 = QLearningAgent(state_dim=2, action_dim=3, n_bins=5)
        agent2.load(save_path)

        assert agent2.epsilon == pytest.approx(original_epsilon)
        assert len(agent2.q_table) == original_q_table_size

        # Check Q-values match
        for key in agent.q_table:
            np.testing.assert_array_almost_equal(
                agent.q_table[key], agent2.q_table[key]
            )

    def test_save_creates_files(self, tmp_path):
        agent = QLearningAgent(state_dim=2, action_dim=2)
        save_path = tmp_path / "q_agent"
        agent.save(save_path)

        assert (save_path / "q_table.pkl").exists()
        assert (save_path / "params.json").exists()


class TestGetInfo:
    """Test get_info method."""

    def test_returns_epsilon(self):
        agent = QLearningAgent(state_dim=2, action_dim=2)
        info = agent.get_info()
        assert "epsilon" in info
        assert info["epsilon"] == 1.0

    def test_returns_q_table_size(self):
        agent = QLearningAgent(state_dim=2, action_dim=2, n_bins=5)
        info = agent.get_info()
        assert "q_table_size" in info
        assert info["q_table_size"] == 0

        # Add entries
        state = np.array([0.5, 0.5])
        agent.train_step(state, 0, 1.0, state, False)
        info = agent.get_info()
        assert info["q_table_size"] > 0
