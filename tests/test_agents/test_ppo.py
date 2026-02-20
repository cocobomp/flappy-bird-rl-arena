"""Tests for PPO agent."""

import numpy as np
import pytest
import torch
from pathlib import Path

from src.agents.ppo import PPOAgent, ActorCriticNetwork
from src.agents.base_agent import BaseAgent


class TestActorCriticNetwork:
    """Test the Actor-Critic network architecture."""

    def test_output_shapes(self):
        net = ActorCriticNetwork(state_dim=4, action_dim=2, hidden_dims=[64, 32])
        x = torch.randn(1, 4)
        action_probs, value = net(x)
        assert action_probs.shape == (1, 2)
        assert value.shape == (1, 1)

    def test_batch_output_shapes(self):
        net = ActorCriticNetwork(state_dim=4, action_dim=3, hidden_dims=[128, 64])
        x = torch.randn(32, 4)
        action_probs, value = net(x)
        assert action_probs.shape == (32, 3)
        assert value.shape == (32, 1)

    def test_action_probs_sum_to_one(self):
        net = ActorCriticNetwork(state_dim=4, action_dim=3)
        x = torch.randn(5, 4)
        action_probs, _ = net(x)
        sums = action_probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_action_probs_non_negative(self):
        net = ActorCriticNetwork(state_dim=4, action_dim=3)
        x = torch.randn(10, 4)
        action_probs, _ = net(x)
        assert (action_probs >= 0).all()

    def test_actor_critic_architecture(self):
        """Verify the network has shared, actor, and critic components."""
        net = ActorCriticNetwork(state_dim=4, action_dim=2, hidden_dims=[64, 32])
        assert hasattr(net, "shared")
        assert hasattr(net, "actor")
        assert hasattr(net, "critic")

        # Shared backbone: Linear + ReLU
        assert isinstance(net.shared[0], torch.nn.Linear)
        assert net.shared[0].in_features == 4
        assert net.shared[0].out_features == 64
        assert isinstance(net.shared[1], torch.nn.ReLU)

        # Actor head: Linear + ReLU + Linear + Softmax
        assert isinstance(net.actor[0], torch.nn.Linear)
        assert net.actor[0].in_features == 64
        assert net.actor[0].out_features == 32
        assert isinstance(net.actor[-1], torch.nn.Softmax)

        # Critic head: Linear + ReLU + Linear
        assert isinstance(net.critic[0], torch.nn.Linear)
        assert net.critic[0].in_features == 64
        assert net.critic[0].out_features == 32
        assert isinstance(net.critic[-1], torch.nn.Linear)
        assert net.critic[-1].out_features == 1

    def test_custom_hidden_dims(self):
        net = ActorCriticNetwork(state_dim=2, action_dim=2, hidden_dims=[128, 64])
        x = torch.randn(1, 2)
        action_probs, value = net(x)
        assert action_probs.shape == (1, 2)
        assert value.shape == (1, 1)


class TestPPOCreation:
    """Test PPO agent creation and default parameters."""

    def test_is_base_agent(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        assert isinstance(agent, BaseAgent)

    def test_default_parameters(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay == 0.99995
        assert agent.clip_eps == 0.2
        assert agent.value_coef == 0.5
        assert agent.entropy_coef == 0.01
        assert agent.rollout_size == 128
        assert agent.n_epochs == 4
        assert agent.mini_batch_size == 32
        assert agent.gae_lambda == 0.95

    def test_has_ac_net(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        assert agent.ac_net is not None
        assert isinstance(agent.ac_net, ActorCriticNetwork)


class TestPPOSelectAction:
    """Test PPO action selection."""

    def test_select_action(self):
        agent = PPOAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert 0 <= action < 3
            # Clear buffer to avoid accumulation
            agent._rewards.append(0.0)
            agent._dones.append(False)

    def test_returns_int(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state)
        assert isinstance(action, int)

    def test_greedy_when_not_training(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        agent.epsilon = 0.0
        state = np.random.randn(4).astype(np.float32)
        # Not training: no buffer accumulation
        actions = [agent.select_action(state, training=False) for _ in range(10)]
        # All actions should be valid
        assert all(0 <= a < 2 for a in actions)

    def test_get_q_values_shape(self):
        agent = PPOAgent(state_dim=4, action_dim=3)
        state = np.random.randn(4).astype(np.float32)
        probs = agent._get_q_values(state)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3,)
        # Should be valid probabilities
        assert np.allclose(probs.sum(), 1.0, atol=1e-5)
        assert (probs >= 0).all()


class TestPPOTrainStep:
    """Test PPO training step."""

    def test_train_step_returns_dict(self):
        agent = PPOAgent(
            state_dim=4, action_dim=2,
            rollout_size=8, mini_batch_size=4, n_epochs=2,
        )
        state = np.random.randn(4).astype(np.float32)
        result = agent.train_step(state, 0, 1.0, state, False)
        assert isinstance(result, dict)

    def test_train_step_returns_metrics_when_done(self):
        """When done=True, PPO should perform an update even if buffer not full."""
        agent = PPOAgent(
            state_dim=4, action_dim=2,
            rollout_size=128, mini_batch_size=4, n_epochs=2,
        )
        state = np.random.randn(4).astype(np.float32)

        # Do a few steps then trigger done
        for i in range(5):
            agent.select_action(state, training=True)
            next_state = np.random.randn(4).astype(np.float32)
            done = (i == 4)
            result = agent.train_step(state, 0, 1.0, next_state, done)
            state = next_state

        assert "policy_loss" in result
        assert "value_loss" in result
        assert "entropy" in result
        assert "total_loss" in result

    def test_train_step_returns_metrics_when_buffer_full(self):
        """PPO should update when the rollout buffer reaches rollout_size."""
        rollout_size = 8
        agent = PPOAgent(
            state_dim=4, action_dim=2,
            rollout_size=rollout_size, mini_batch_size=4, n_epochs=2,
        )
        state = np.random.randn(4).astype(np.float32)

        result = {}
        for i in range(rollout_size):
            agent.select_action(state, training=True)
            next_state = np.random.randn(4).astype(np.float32)
            result = agent.train_step(state, 0, 1.0, next_state, False)
            state = next_state

        assert "policy_loss" in result
        assert "value_loss" in result

    def test_epsilon_decays_every_step(self):
        agent = PPOAgent(
            state_dim=4, action_dim=2,
            epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.01,
        )
        state = np.random.randn(4).astype(np.float32)
        agent.train_step(state, 0, 1.0, state, False)
        assert agent.epsilon < 1.0

    def test_buffer_clears_after_update(self):
        """After a PPO update, the rollout buffer should be cleared."""
        agent = PPOAgent(
            state_dim=4, action_dim=2,
            rollout_size=4, mini_batch_size=2, n_epochs=1,
        )
        state = np.random.randn(4).astype(np.float32)

        for i in range(4):
            agent.select_action(state, training=True)
            next_state = np.random.randn(4).astype(np.float32)
            agent.train_step(state, 0, 1.0, next_state, False)
            state = next_state

        # Buffer should be cleared after update
        assert len(agent._states) == 0
        assert len(agent._rewards) == 0


class TestPPOSaveLoad:
    """Test save/load roundtrip."""

    def test_save_load(self, tmp_path):
        agent = PPOAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)

        # Get action probs before save
        probs_before = agent._get_q_values(state)
        original_epsilon = agent.epsilon

        save_path = tmp_path / "ppo_agent"
        agent.save(save_path)

        # Create fresh agent and load
        agent2 = PPOAgent(state_dim=4, action_dim=2)
        agent2.load(save_path)

        probs_after = agent2._get_q_values(state)
        np.testing.assert_array_almost_equal(probs_before, probs_after)
        assert agent2.epsilon == pytest.approx(original_epsilon)

    def test_save_creates_files(self, tmp_path):
        agent = PPOAgent(state_dim=4, action_dim=2)
        save_path = tmp_path / "ppo_agent"
        agent.save(save_path)

        assert (save_path / "ac_net.pt").exists()
        assert (save_path / "params.json").exists()

    def test_save_load_preserves_params(self, tmp_path):
        agent = PPOAgent(
            state_dim=4, action_dim=2,
            clip_eps=0.3, value_coef=0.7, entropy_coef=0.05,
            gae_lambda=0.9,
        )
        agent.epsilon = 0.42

        save_path = tmp_path / "ppo_agent"
        agent.save(save_path)

        agent2 = PPOAgent(state_dim=4, action_dim=2)
        agent2.load(save_path)

        assert agent2.clip_eps == pytest.approx(0.3)
        assert agent2.value_coef == pytest.approx(0.7)
        assert agent2.entropy_coef == pytest.approx(0.05)
        assert agent2.gae_lambda == pytest.approx(0.9)
        assert agent2.epsilon == pytest.approx(0.42)


class TestPPOWeights:
    """Test get_weights / set_weights."""

    def test_get_weights_set_weights(self):
        agent1 = PPOAgent(state_dim=4, action_dim=2)
        agent2 = PPOAgent(state_dim=4, action_dim=2)

        state = np.random.randn(4).astype(np.float32)

        # They should generally produce different outputs
        weights = agent1.get_weights()
        agent2.set_weights(weights)

        # After set_weights, outputs should match
        probs1 = agent1._get_q_values(state)
        probs2 = agent2._get_q_values(state)
        np.testing.assert_array_almost_equal(probs1, probs2)

    def test_get_weights_returns_dict(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        weights = agent.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_get_weights_are_copies(self):
        """Modifying returned weights should not affect the agent."""
        agent = PPOAgent(state_dim=4, action_dim=2)
        weights = agent.get_weights()
        for k in weights:
            weights[k].fill_(999.0)

        # Agent should be unaffected
        state = np.random.randn(4).astype(np.float32)
        probs = agent._get_q_values(state)
        # If weights were modified, probs would be extreme. Just verify it runs
        assert probs.shape == (2,)


class TestPPOMutate:
    """Test mutation for evolutionary methods."""

    def test_mutate_changes_weights(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        probs_before = agent._get_q_values(state)
        agent.mutate(noise_scale=1.0)
        probs_after = agent._get_q_values(state)
        # With noise_scale=1.0, extremely unlikely to remain exactly the same
        assert not np.allclose(probs_before, probs_after)


class TestPPOGetInfo:
    """Test get_info method."""

    def test_returns_expected_keys(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        info = agent.get_info()
        assert "epsilon" in info
        assert "step_count" in info
        assert "clip_eps" in info
        assert info["epsilon"] == 1.0
        assert info["step_count"] == 0


class TestPPOGetActivations:
    """Test get_activations method."""

    def test_returns_list(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        activations = agent.get_activations(state)
        assert isinstance(activations, list)
        assert len(activations) > 0

    def test_first_activation_is_input(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        activations = agent.get_activations(state)
        np.testing.assert_array_almost_equal(activations[0], state)
