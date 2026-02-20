"""Tests for DQN agent."""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.agents.dqn import DQNAgent, QNetwork, ReplayBuffer
from src.agents.base_agent import BaseAgent


class TestQNetwork:
    """Test the Q-Network architecture."""

    def test_output_shape(self):
        net = QNetwork(state_dim=4, action_dim=2, hidden_dims=[64, 64])
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_batch_output_shape(self):
        net = QNetwork(state_dim=4, action_dim=3, hidden_dims=[128, 128])
        x = torch.randn(32, 4)
        out = net(x)
        assert out.shape == (32, 3)

    def test_custom_hidden_dims(self):
        net = QNetwork(state_dim=2, action_dim=2, hidden_dims=[32, 16, 8])
        x = torch.randn(1, 2)
        out = net(x)
        assert out.shape == (1, 2)

    def test_single_hidden_layer(self):
        net = QNetwork(state_dim=4, action_dim=2, hidden_dims=[64])
        x = torch.randn(5, 4)
        out = net(x)
        assert out.shape == (5, 2)


class TestReplayBuffer:
    """Test the experience replay buffer."""

    def test_push_and_len(self):
        buf = ReplayBuffer(max_size=100)
        assert len(buf) == 0
        buf.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert len(buf) == 1

    def test_max_size(self):
        buf = ReplayBuffer(max_size=5)
        for i in range(10):
            buf.push(np.zeros(4), 0, float(i), np.zeros(4), False)
        assert len(buf) == 5

    def test_sample_shape(self):
        buf = ReplayBuffer(max_size=100)
        state_dim = 4
        for _ in range(20):
            buf.push(
                np.random.randn(state_dim), 1, 0.5,
                np.random.randn(state_dim), False,
            )

        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, state_dim)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, state_dim)
        assert dones.shape == (8,)

    def test_sample_returns_numpy(self):
        buf = ReplayBuffer(max_size=100)
        for _ in range(10):
            buf.push(np.zeros(2), 0, 1.0, np.zeros(2), False)
        states, actions, rewards, next_states, dones = buf.sample(5)
        assert isinstance(states, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(next_states, np.ndarray)
        assert isinstance(dones, np.ndarray)

    def test_sample_content(self):
        buf = ReplayBuffer(max_size=10)
        buf.push(np.array([1.0, 2.0]), 1, 3.0, np.array([4.0, 5.0]), True)
        states, actions, rewards, next_states, dones = buf.sample(1)
        np.testing.assert_array_almost_equal(states[0], [1.0, 2.0])
        assert actions[0] == 1
        assert rewards[0] == pytest.approx(3.0)
        np.testing.assert_array_almost_equal(next_states[0], [4.0, 5.0])
        assert dones[0] == 1.0  # True -> 1.0


class TestDQNCreation:
    """Test DQN agent creation and default parameters."""

    def test_is_base_agent_subclass(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        assert isinstance(agent, BaseAgent)

    def test_default_parameters(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.gamma == 0.99
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay == 0.995
        assert agent.batch_size == 64
        assert agent.tau == 0.005
        assert agent.train_every == 4

    def test_has_q_net_and_target_net(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        assert agent.q_net is not None
        assert agent.target_net is not None

    def test_networks_on_same_device(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        q_device = next(agent.q_net.parameters()).device
        t_device = next(agent.target_net.parameters()).device
        assert q_device == t_device


class TestDQNSelectAction:
    """Test DQN action selection."""

    def test_returns_valid_action(self):
        agent = DQNAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert 0 <= action < 3

    def test_greedy_with_epsilon_zero(self):
        agent = DQNAgent(state_dim=4, action_dim=3)
        agent.epsilon = 0.0
        state = np.random.randn(4).astype(np.float32)
        # With epsilon=0, should always return same (greedy) action
        actions = [agent.select_action(state, training=True) for _ in range(50)]
        assert len(set(actions)) == 1

    def test_greedy_when_not_training(self):
        agent = DQNAgent(state_dim=4, action_dim=3)
        agent.epsilon = 1.0  # high epsilon
        state = np.random.randn(4).astype(np.float32)
        # training=False should always pick greedy
        actions = [agent.select_action(state, training=False) for _ in range(50)]
        assert len(set(actions)) == 1

    def test_get_q_values_shape(self):
        agent = DQNAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        q_vals = agent._get_q_values(state)
        assert isinstance(q_vals, np.ndarray)
        assert q_vals.shape == (3,)


class TestDQNTrainStep:
    """Test DQN training step."""

    def test_returns_dict(self):
        agent = DQNAgent(state_dim=4, action_dim=2, buffer_size=100, batch_size=4)
        state = np.random.randn(4).astype(np.float32)
        result = agent.train_step(state, 0, 1.0, state, False)
        assert isinstance(result, dict)

    def test_returns_loss_after_enough_samples(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2,
            buffer_size=200, batch_size=8, train_every=1,
        )
        # Fill the buffer past batch_size
        state = np.random.randn(4).astype(np.float32)
        for i in range(20):
            next_state = np.random.randn(4).astype(np.float32)
            result = agent.train_step(state, i % 2, 1.0, next_state, False)
            state = next_state

        # After enough samples, should have computed a loss
        assert "loss" in result
        assert "q_mean" in result
        assert isinstance(result["loss"], float)

    def test_epsilon_decays_on_train_step(self):
        agent = DQNAgent(
            state_dim=4, action_dim=2,
            epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.01,
        )
        state = np.random.randn(4).astype(np.float32)
        agent.train_step(state, 0, 1.0, state, False)
        assert agent.epsilon == pytest.approx(0.5)

    def test_buffer_grows(self):
        agent = DQNAgent(state_dim=4, action_dim=2, buffer_size=100)
        state = np.random.randn(4).astype(np.float32)
        assert len(agent.replay_buffer) == 0
        agent.train_step(state, 0, 1.0, state, False)
        assert len(agent.replay_buffer) == 1


class TestDQNTargetUpdate:
    """Test target network soft update."""

    def test_soft_update_with_tau_one(self):
        """With tau=1.0, target should exactly match online after update."""
        agent = DQNAgent(state_dim=4, action_dim=2, tau=1.0)

        # Manually change online network
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.fill_(42.0)

        agent._update_target()

        # Target should now be identical
        for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(qp, tp)

    def test_soft_update_with_tau_zero(self):
        """With tau=0.0, target should not change."""
        agent = DQNAgent(state_dim=4, action_dim=2, tau=0.0)

        # Record original target params
        original = [p.clone() for p in agent.target_net.parameters()]

        # Change online network
        with torch.no_grad():
            for p in agent.q_net.parameters():
                p.fill_(999.0)

        agent._update_target()

        # Target should be unchanged
        for orig, tp in zip(original, agent.target_net.parameters()):
            assert torch.allclose(orig, tp)


class TestDQNSaveLoad:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        agent = DQNAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)

        # Get Q-values before save
        q_before = agent._get_q_values(state)
        original_epsilon = agent.epsilon

        save_path = tmp_path / "dqn_agent"
        agent.save(save_path)

        # Create fresh agent and load
        agent2 = DQNAgent(state_dim=4, action_dim=2)
        agent2.load(save_path)

        q_after = agent2._get_q_values(state)
        np.testing.assert_array_almost_equal(q_before, q_after)
        assert agent2.epsilon == pytest.approx(original_epsilon)

    def test_save_creates_files(self, tmp_path):
        agent = DQNAgent(state_dim=4, action_dim=2)
        save_path = tmp_path / "dqn_agent"
        agent.save(save_path)

        assert (save_path / "q_net.pt").exists()
        assert (save_path / "target_net.pt").exists()
        assert (save_path / "params.json").exists()


class TestDQNGetInfo:
    """Test get_info method."""

    def test_returns_expected_keys(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        info = agent.get_info()
        assert "epsilon" in info
        assert info["epsilon"] == 1.0
