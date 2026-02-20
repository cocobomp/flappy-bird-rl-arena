"""Tests for BaseAgent abstract interface."""

import numpy as np
import pytest
from pathlib import Path

from src.agents.base_agent import BaseAgent


class TestBaseAgentCannotBeInstantiated:
    """BaseAgent is abstract and cannot be instantiated directly."""

    def test_direct_instantiation_raises_type_error(self):
        with pytest.raises(TypeError):
            BaseAgent(state_dim=4, action_dim=2)


class TestIncompleteSubclassFails:
    """A subclass that does not implement all abstract methods cannot be instantiated."""

    def test_missing_all_methods(self):
        class EmptyAgent(BaseAgent):
            pass

        with pytest.raises(TypeError):
            EmptyAgent(state_dim=4, action_dim=2)

    def test_missing_some_methods(self):
        class PartialAgent(BaseAgent):
            def select_action(self, state, training=True):
                return 0

            def train_step(self, state, action, reward, next_state, done):
                return {}

            # Missing save, load, get_info

        with pytest.raises(TypeError):
            PartialAgent(state_dim=4, action_dim=2)

    def test_missing_only_get_info(self):
        class AlmostAgent(BaseAgent):
            def select_action(self, state, training=True):
                return 0

            def train_step(self, state, action, reward, next_state, done):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

            # Missing get_info

        with pytest.raises(TypeError):
            AlmostAgent(state_dim=4, action_dim=2)


class TestCompleteSubclass:
    """A complete subclass should work fine."""

    def _make_complete_agent(self, state_dim=4, action_dim=2):
        class ConcreteAgent(BaseAgent):
            def select_action(self, state, training=True):
                return 0

            def train_step(self, state, action, reward, next_state, done):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

            def get_info(self):
                return {}

        return ConcreteAgent(state_dim=state_dim, action_dim=action_dim)

    def test_complete_subclass_instantiates(self):
        agent = self._make_complete_agent()
        assert agent is not None

    def test_state_dim_stored(self):
        agent = self._make_complete_agent(state_dim=8, action_dim=3)
        assert agent.state_dim == 8

    def test_action_dim_stored(self):
        agent = self._make_complete_agent(state_dim=8, action_dim=3)
        assert agent.action_dim == 3

    def test_select_action_returns_int(self):
        agent = self._make_complete_agent()
        result = agent.select_action(np.zeros(4))
        assert isinstance(result, int)

    def test_train_step_returns_dict(self):
        agent = self._make_complete_agent()
        result = agent.train_step(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert isinstance(result, dict)

    def test_get_info_returns_dict(self):
        agent = self._make_complete_agent()
        result = agent.get_info()
        assert isinstance(result, dict)
