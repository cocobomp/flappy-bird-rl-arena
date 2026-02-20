"""Tests for REINFORCE (Policy Gradient) agent."""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.agents.reinforce import ReinforceAgent, PolicyNetwork
from src.agents.base_agent import BaseAgent


class TestPolicyNetwork:
    """Test the Policy Network architecture."""

    def test_output_shape(self):
        net = PolicyNetwork(state_dim=4, action_dim=2, hidden_dims=[64, 64])
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_batch_output_shape(self):
        net = PolicyNetwork(state_dim=4, action_dim=3, hidden_dims=[128, 128])
        x = torch.randn(32, 4)
        out = net(x)
        assert out.shape == (32, 3)

    def test_output_sums_to_one(self):
        """Policy network outputs should be valid probability distributions."""
        net = PolicyNetwork(state_dim=4, action_dim=3)
        x = torch.randn(5, 4)
        out = net(x)
        sums = out.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_output_non_negative(self):
        """All probabilities should be non-negative."""
        net = PolicyNetwork(state_dim=4, action_dim=3)
        x = torch.randn(10, 4)
        out = net(x)
        assert (out >= 0).all()

    def test_custom_hidden_dims(self):
        net = PolicyNetwork(state_dim=2, action_dim=2, hidden_dims=[32, 16, 8])
        x = torch.randn(1, 2)
        out = net(x)
        assert out.shape == (1, 2)

    def test_single_hidden_layer(self):
        net = PolicyNetwork(state_dim=4, action_dim=2, hidden_dims=[64])
        x = torch.randn(5, 4)
        out = net(x)
        assert out.shape == (5, 2)


class TestReinforceCreation:
    """Test REINFORCE agent creation and default parameters."""

    def test_is_base_agent(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        assert isinstance(agent, BaseAgent)

    def test_default_parameters(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay == 0.99995
        assert agent.lr == 3e-4

    def test_has_policy_net(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        assert agent.policy_net is not None
        assert isinstance(agent.policy_net, PolicyNetwork)

    def test_custom_parameters(self):
        agent = ReinforceAgent(
            state_dim=8,
            action_dim=3,
            hidden_dims=[128, 64],
            lr=1e-3,
            gamma=0.99,
            epsilon_start=0.5,
            epsilon_end=0.05,
            epsilon_decay=0.999,
        )
        assert agent.state_dim == 8
        assert agent.action_dim == 3
        assert agent.gamma == 0.99
        assert agent.epsilon == 0.5
        assert agent.epsilon_end == 0.05
        assert agent.epsilon_decay == 0.999
        assert agent.lr == 1e-3

    def test_episode_buffer_starts_empty(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        assert len(agent._log_probs) == 0
        assert len(agent._rewards) == 0


class TestReinforceSelectAction:
    """Test REINFORCE action selection."""

    def test_select_action_valid(self):
        agent = ReinforceAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert 0 <= action < 3

    def test_greedy_when_not_training(self):
        agent = ReinforceAgent(state_dim=4, action_dim=3)
        agent.epsilon = 1.0  # high epsilon
        state = np.random.randn(4).astype(np.float32)
        # training=False should always pick greedy (argmax of probs)
        actions = [agent.select_action(state, training=False) for _ in range(50)]
        assert len(set(actions)) == 1

    def test_get_q_values_shape(self):
        """_get_q_values returns action probabilities for display."""
        agent = ReinforceAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        probs = agent._get_q_values(state)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3,)

    def test_get_q_values_sums_to_one(self):
        """Probabilities from _get_q_values should sum to 1."""
        agent = ReinforceAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        probs = agent._get_q_values(state)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_get_activations(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2, hidden_dims=[16, 8])
        state = np.random.randn(4).astype(np.float32)
        activations = agent.get_activations(state)
        # Should have: input (4,), hidden1 (16,), hidden2 (8,), output (2,)
        assert len(activations) >= 3
        assert activations[0].shape == (4,)
        assert activations[-1].shape == (2,)


class TestReinforceTrainStep:
    """Test REINFORCE training step."""

    def test_train_step_returns_dict(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        result = agent.train_step(state, 0, 1.0, state, False)
        assert isinstance(result, dict)

    def test_no_loss_before_episode_end(self):
        """Should not train (no loss) until the episode ends."""
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        agent.epsilon = 0.0  # no random exploration
        state = np.random.randn(4).astype(np.float32)
        agent.select_action(state, training=True)
        result = agent.train_step(state, 0, 1.0, state, False)
        assert "loss" not in result

    def test_trains_on_episode_end(self):
        """Should compute and return loss when done=True."""
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        agent.epsilon = 0.0  # disable exploration for deterministic log_probs

        # Simulate a short episode
        states = [np.random.randn(4).astype(np.float32) for _ in range(5)]

        for i in range(4):
            action = agent.select_action(states[i], training=True)
            agent.train_step(states[i], action, 1.0, states[i + 1], False)

        # Final step: done=True triggers training
        action = agent.select_action(states[4], training=True)
        result = agent.train_step(states[4], action, -1.0, states[4], True)

        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_buffer_cleared_after_training(self):
        """Episode buffer should be empty after done=True training."""
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        agent.epsilon = 0.0

        state = np.random.randn(4).astype(np.float32)
        for _ in range(3):
            action = agent.select_action(state, training=True)
            agent.train_step(state, action, 1.0, state, False)

        action = agent.select_action(state, training=True)
        agent.train_step(state, action, 1.0, state, True)  # episode ends

        assert len(agent._log_probs) == 0
        assert len(agent._rewards) == 0

    def test_epsilon_decays_every_step(self):
        """Epsilon should decay on every call to train_step."""
        agent = ReinforceAgent(
            state_dim=4,
            action_dim=2,
            epsilon_start=1.0,
            epsilon_decay=0.5,
            epsilon_end=0.01,
        )
        state = np.random.randn(4).astype(np.float32)
        agent.select_action(state, training=True)
        agent.train_step(state, 0, 1.0, state, False)
        assert agent.epsilon < 1.0

    def test_multiple_episodes(self):
        """Agent should handle multiple episodes correctly."""
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        agent.epsilon = 0.0

        for episode in range(3):
            state = np.random.randn(4).astype(np.float32)
            for step in range(5):
                action = agent.select_action(state, training=True)
                done = step == 4
                result = agent.train_step(state, action, 1.0, state, done)
                if done:
                    assert "loss" in result
                    assert len(agent._log_probs) == 0
                    assert len(agent._rewards) == 0


class TestReinforceSaveLoad:
    """Test save/load roundtrip."""

    def test_save_load(self, tmp_path):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)

        # Get probabilities before save
        probs_before = agent._get_q_values(state)
        original_epsilon = agent.epsilon

        save_path = tmp_path / "reinforce_agent"
        agent.save(save_path)

        # Create fresh agent and load
        agent2 = ReinforceAgent(state_dim=4, action_dim=2)
        agent2.load(save_path)

        probs_after = agent2._get_q_values(state)
        np.testing.assert_array_almost_equal(probs_before, probs_after)
        assert agent2.epsilon == pytest.approx(original_epsilon)

    def test_save_creates_files(self, tmp_path):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        save_path = tmp_path / "reinforce_agent"
        agent.save(save_path)

        assert (save_path / "policy_net.pt").exists()
        assert (save_path / "params.json").exists()


class TestReinforceGetInfo:
    """Test get_info method."""

    def test_returns_expected_keys(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        info = agent.get_info()
        assert "epsilon" in info
        assert "step_count" in info
        assert info["epsilon"] == 1.0
        assert info["step_count"] == 0


class TestReinforceWeights:
    """Test get_weights, set_weights, and mutate."""

    def test_get_weights_set_weights(self):
        agent1 = ReinforceAgent(state_dim=4, action_dim=2)
        agent2 = ReinforceAgent(state_dim=4, action_dim=2)

        state = np.random.randn(4).astype(np.float32)

        # Agents should differ initially (random init)
        probs1 = agent1._get_q_values(state)
        probs2 = agent2._get_q_values(state)

        # Copy weights from agent1 to agent2
        weights = agent1.get_weights()
        agent2.set_weights(weights)

        # Now they should match
        probs1_after = agent1._get_q_values(state)
        probs2_after = agent2._get_q_values(state)
        np.testing.assert_array_almost_equal(probs1_after, probs2_after)

    def test_mutate_changes_weights(self):
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)

        probs_before = agent._get_q_values(state)
        agent.mutate(noise_scale=1.0)  # large noise to ensure change
        probs_after = agent._get_q_values(state)

        # With large noise, probabilities should differ
        assert not np.allclose(probs_before, probs_after)

    def test_get_weights_returns_copies(self):
        """get_weights should return cloned tensors, not references."""
        agent = ReinforceAgent(state_dim=4, action_dim=2)
        weights = agent.get_weights()

        # Modify the returned weights
        for k in weights:
            weights[k].fill_(999.0)

        # Agent's weights should be unaffected
        current = agent.get_weights()
        for k in current:
            assert not (current[k] == 999.0).all()
