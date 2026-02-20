"""Tests for Dueling DQN agent."""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.agents.dueling_dqn import DuelingDQNAgent, DuelingQNetwork
from src.agents.dqn import DQNAgent
from src.agents.base_agent import BaseAgent


class TestDuelingDQNCreation:
    """Test Dueling DQN creation and inheritance."""

    def test_is_dqn_subclass(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        assert isinstance(agent, DQNAgent)

    def test_is_base_agent_subclass(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        assert isinstance(agent, BaseAgent)

    def test_default_parameters(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0

    def test_custom_parameters(self):
        agent = DuelingDQNAgent(
            state_dim=8, action_dim=3,
            hidden_dims=[128, 64], lr=1e-3, gamma=0.99,
        )
        assert agent.state_dim == 8
        assert agent.action_dim == 3
        assert agent.gamma == 0.99


class TestDuelingArchitecture:
    """Test that the dueling architecture has the correct structure."""

    def test_has_shared_layer(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        assert hasattr(agent.q_net, "shared")

    def test_has_value_stream(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        assert hasattr(agent.q_net, "value_stream")

    def test_has_advantage_stream(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        assert hasattr(agent.q_net, "advantage_stream")

    def test_value_stream_outputs_scalar(self):
        net = DuelingQNetwork(state_dim=4, action_dim=2)
        x = torch.randn(1, 4)
        features = net.shared(x)
        value = net.value_stream(features)
        assert value.shape == (1, 1)

    def test_advantage_stream_outputs_action_dim(self):
        net = DuelingQNetwork(state_dim=4, action_dim=3)
        x = torch.randn(1, 4)
        features = net.shared(x)
        advantage = net.advantage_stream(features)
        assert advantage.shape == (1, 3)

    def test_q_net_is_dueling(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        assert isinstance(agent.q_net, DuelingQNetwork)
        assert isinstance(agent.target_net, DuelingQNetwork)

    def test_dueling_decomposition(self):
        """Verify Q = V + A - mean(A) by checking the math."""
        net = DuelingQNetwork(state_dim=4, action_dim=3)
        x = torch.randn(2, 4)

        with torch.no_grad():
            features = net.shared(x)
            value = net.value_stream(features)        # (2, 1)
            advantage = net.advantage_stream(features)  # (2, 3)

            expected_q = value + advantage - advantage.mean(dim=1, keepdim=True)
            actual_q = net(x)

        np.testing.assert_array_almost_equal(
            expected_q.numpy(), actual_q.numpy()
        )

    def test_hidden_dims_too_short_raises(self):
        """Dueling DQN requires at least 2 hidden dims."""
        with pytest.raises(ValueError, match="at least 2 hidden_dims"):
            DuelingQNetwork(state_dim=4, action_dim=2, hidden_dims=[64])


class TestDuelingDQNOutputShape:
    """Test that the network produces the right output shapes."""

    def test_output_shape_single_state(self):
        net = DuelingQNetwork(state_dim=4, action_dim=2)
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_output_shape_batch(self):
        net = DuelingQNetwork(state_dim=4, action_dim=3)
        x = torch.randn(32, 4)
        out = net(x)
        assert out.shape == (32, 3)

    def test_q_values_shape(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        q_values = agent._get_q_values(state)
        assert q_values.shape == (2,)


class TestDuelingDQNSelectAction:
    """Test that action selection works correctly."""

    def test_returns_valid_action(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert 0 <= action < 3

    def test_greedy_with_epsilon_zero(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=3)
        agent.epsilon = 0.0
        state = np.random.randn(4).astype(np.float32)
        actions = [agent.select_action(state, training=True) for _ in range(50)]
        assert len(set(actions)) == 1


class TestDuelingDQNTrainStep:
    """Test Dueling DQN training step."""

    def test_returns_dict(self):
        agent = DuelingDQNAgent(
            state_dim=4, action_dim=2, buffer_size=100, batch_size=4,
        )
        state = np.random.randn(4).astype(np.float32)
        result = agent.train_step(state, 0, 1.0, state, False)
        assert isinstance(result, dict)

    def test_returns_loss_after_enough_samples(self):
        agent = DuelingDQNAgent(
            state_dim=4, action_dim=2,
            buffer_size=200, batch_size=8, train_every=1,
        )
        state = np.random.randn(4).astype(np.float32)
        for i in range(20):
            next_state = np.random.randn(4).astype(np.float32)
            result = agent.train_step(state, i % 2, 1.0, next_state, False)
            state = next_state

        assert "loss" in result
        assert "q_mean" in result
        assert isinstance(result["loss"], float)

    def test_epsilon_decays_every_step(self):
        agent = DuelingDQNAgent(
            state_dim=4, action_dim=2,
            epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.01,
            batch_size=4, buffer_size=100, train_every=1,
        )
        state = np.random.randn(4).astype(np.float32)
        agent.train_step(state, 0, 1.0, state, False)
        assert agent.epsilon < 1.0

    def test_training_updates_weights(self):
        """Verify that training actually changes the network weights."""
        agent = DuelingDQNAgent(
            state_dim=4, action_dim=2,
            buffer_size=200, batch_size=8, train_every=1,
        )

        # Capture initial weights
        initial_weights = {
            k: v.clone() for k, v in agent.q_net.state_dict().items()
        }

        # Run enough training steps
        state = np.random.randn(4).astype(np.float32)
        for i in range(30):
            next_state = np.random.randn(4).astype(np.float32)
            agent.train_step(state, i % 2, 1.0, next_state, False)
            state = next_state

        # Check that at least some weights changed
        changed = False
        for k, v in agent.q_net.state_dict().items():
            if not torch.equal(v, initial_weights[k]):
                changed = True
                break
        assert changed, "Training should change network weights"


class TestDuelingDQNSaveLoad:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        q_before = agent._get_q_values(state)

        save_path = tmp_path / "dueling_agent"
        agent.save(save_path)

        agent2 = DuelingDQNAgent(state_dim=4, action_dim=2)
        agent2.load(save_path)

        q_after = agent2._get_q_values(state)
        np.testing.assert_array_almost_equal(q_before, q_after)

    def test_save_load_preserves_epsilon(self, tmp_path):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        agent.epsilon = 0.42

        save_path = tmp_path / "dueling_agent"
        agent.save(save_path)

        agent2 = DuelingDQNAgent(state_dim=4, action_dim=2)
        agent2.load(save_path)

        assert abs(agent2.epsilon - 0.42) < 1e-6


class TestDuelingDQNGetActivations:
    """Test get_activations for visualization support."""

    def test_returns_list(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        activations = agent.get_activations(state)
        assert isinstance(activations, list)

    def test_first_is_input(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        activations = agent.get_activations(state)
        np.testing.assert_array_equal(activations[0], state)

    def test_last_is_output(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        activations = agent.get_activations(state)
        q_values = agent._get_q_values(state)
        np.testing.assert_array_almost_equal(activations[-1], q_values)

    def test_shared_layer_activations(self):
        """Activations should include shared layer output (64 neurons by default)."""
        agent = DuelingDQNAgent(state_dim=4, action_dim=2, hidden_dims=[64, 32])
        state = np.random.randn(4).astype(np.float32)
        activations = agent.get_activations(state)
        # activations: [input(4), shared_relu(64), output(2)]
        assert len(activations) == 3
        assert activations[1].shape == (64,)
        assert activations[2].shape == (2,)


class TestDuelingDQNGetInfo:
    """Test get_info (inherited from DQN)."""

    def test_returns_expected_keys(self):
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        info = agent.get_info()
        assert "epsilon" in info
        assert "step_count" in info
