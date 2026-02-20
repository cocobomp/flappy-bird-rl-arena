# RL Flappy Bird — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a visual demo comparing Q-Learning, DQN, and Double DQN on Flappy Bird with reward shaping, observation, and hyperparameter variations.

**Architecture:** 4 parallel modules (environments, agents, training, visualization) sharing interfaces via `BaseAgent` ABC and YAML configs. Uses `flappy-bird-gymnasium` as base environment with custom wrappers.

**Tech Stack:** Python 3.11+, PyTorch, Gymnasium, flappy-bird-gymnasium, Pygame, NumPy, PyYAML

---

## Parallel Tracks

These 4 tracks can be developed simultaneously by separate agents:

| Track | Agent | Tasks | Dependencies |
|---|---|---|---|
| A — Project Setup & Environment | Agent Env | 1-3 | None |
| B — RL Algorithms | Agent Algos | 4-7 | None (uses BaseAgent interface defined in Task 4) |
| C — Training Pipeline | Agent Training | 8-10 | After Tasks 3 & 7 (needs wrappers + agents) |
| D — Visualization | Agent Visu | 11-13 | After Tasks 3 & 7 (needs wrappers + agents) |
| E — Integration | All | 14 | After all tracks |

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/environments/__init__.py`
- Create: `src/agents/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/visualization/__init__.py`
- Create: `configs/`
- Create: `models/`
- Create: `tests/__init__.py`
- Create: `tests/test_environments/__init__.py`
- Create: `tests/test_agents/__init__.py`
- Create: `tests/test_training/__init__.py`

**Step 1: Create requirements.txt**

```
flappy-bird-gymnasium>=0.4.0
gymnasium>=0.29.0
pygame>=2.5.0
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
```

**Step 2: Create directory structure**

```bash
mkdir -p src/environments src/agents src/training src/visualization configs models tests/test_environments tests/test_agents tests/test_training
```

**Step 3: Create all __init__.py files**

Empty `__init__.py` in each package directory.

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Verify installation**

```python
import flappy_bird_gymnasium
import gymnasium
import torch
import pygame
env = gymnasium.make("FlappyBird-v0")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
env.close()
```

Expected: prints observation shape (12,) for default mode.

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffold project structure and dependencies"
```

---

### Task 2: Reward functions

**Files:**
- Create: `src/environments/rewards.py`
- Create: `tests/test_environments/test_rewards.py`

**Step 1: Write failing tests**

```python
# tests/test_environments/test_rewards.py
import numpy as np
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward


class TestBasicReward:
    def test_alive_reward(self):
        reward_fn = BasicReward()
        # Not dead => +1
        assert reward_fn.compute(
            obs=np.array([0.5, 0.0, 0.3, 0.1, 0.5, 0.7, 0.8, 0.2, 0.6, 0.4, 0.5, 0.0]),
            raw_reward=0.1,
            terminated=False,
            truncated=False,
        ) == 1.0

    def test_death_penalty(self):
        reward_fn = BasicReward()
        assert reward_fn.compute(
            obs=np.array([0.5, 0.0, 0.3, 0.1, 0.5, 0.7, 0.8, 0.2, 0.6, 0.4, 0.5, 0.0]),
            raw_reward=-1.0,
            terminated=True,
            truncated=False,
        ) == -1000.0


class TestDistanceReward:
    def test_reward_proportional_to_pipe_proximity(self):
        reward_fn = DistanceReward()
        # next_pipe_x is obs[2] (horizontal position of next pipe)
        # Closer to pipe => higher reward
        obs_close = np.array([0.5, 0.0, 0.1, 0.1, 0.5, 0.7, 0.8, 0.2, 0.6, 0.4, 0.5, 0.0])
        obs_far = np.array([0.5, 0.0, 0.9, 0.1, 0.5, 0.7, 0.8, 0.2, 0.6, 0.4, 0.5, 0.0])
        r_close = reward_fn.compute(obs=obs_close, raw_reward=0.1, terminated=False, truncated=False)
        r_far = reward_fn.compute(obs=obs_far, raw_reward=0.1, terminated=False, truncated=False)
        assert r_close > r_far

    def test_death_still_penalized(self):
        reward_fn = DistanceReward()
        r = reward_fn.compute(
            obs=np.zeros(12), raw_reward=-1.0, terminated=True, truncated=False
        )
        assert r < 0


class TestCenteredReward:
    def test_centered_bird_gets_bonus(self):
        reward_fn = CenteredReward()
        # player_y is obs[9], next_pipe top is obs[3], next_pipe bottom is obs[5]
        # gap center = (obs[3] + obs[5]) / 2
        # bird at gap center => max bonus
        gap_center = (0.3 + 0.7) / 2  # 0.5
        obs_centered = np.array([0.5, 0.0, 0.2, 0.3, 0.2, 0.7, 0.8, 0.2, 0.6, gap_center, 0.5, 0.0])
        obs_off = np.array([0.5, 0.0, 0.2, 0.3, 0.2, 0.7, 0.8, 0.2, 0.6, 0.1, 0.5, 0.0])
        r_centered = reward_fn.compute(obs=obs_centered, raw_reward=0.1, terminated=False, truncated=False)
        r_off = reward_fn.compute(obs=obs_off, raw_reward=0.1, terminated=False, truncated=False)
        assert r_centered > r_off
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_environments/test_rewards.py -v
```

Expected: FAIL (module not found)

**Step 3: Implement reward functions**

```python
# src/environments/rewards.py
from abc import ABC, abstractmethod
import numpy as np


class RewardFunction(ABC):
    @abstractmethod
    def compute(
        self,
        obs: np.ndarray,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float:
        pass


class BasicReward(RewardFunction):
    """Simple survival reward: +1 alive, -1000 dead."""

    def compute(self, obs, raw_reward, terminated, truncated):
        if terminated:
            return -1000.0
        return 1.0


class DistanceReward(RewardFunction):
    """Reward proportional to proximity to next pipe (progress)."""

    def compute(self, obs, raw_reward, terminated, truncated):
        if terminated:
            return -1000.0
        # obs[2] = horizontal distance to next pipe (normalized, smaller = closer)
        progress = 1.0 - float(obs[2])
        return progress


class CenteredReward(RewardFunction):
    """Bonus for staying centered in the pipe gap."""

    def compute(self, obs, raw_reward, terminated, truncated):
        if terminated:
            return -1000.0
        # obs[3] = next pipe top y, obs[5] = next pipe bottom y
        # obs[9] = player y
        gap_center = (float(obs[3]) + float(obs[5])) / 2.0
        player_y = float(obs[9])
        distance_to_center = abs(player_y - gap_center)
        # Max bonus 2.0 when perfectly centered, min 0.0
        bonus = max(0.0, 2.0 - 4.0 * distance_to_center)
        return 1.0 + bonus
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_environments/test_rewards.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/environments/rewards.py tests/test_environments/test_rewards.py
git commit -m "feat: add reward functions (basic, distance, centered)"
```

---

### Task 3: Environment wrappers

**Files:**
- Create: `src/environments/wrappers.py`
- Create: `tests/test_environments/test_wrappers.py`

**Step 1: Write failing tests**

```python
# tests/test_environments/test_wrappers.py
import gymnasium
import numpy as np
import flappy_bird_gymnasium
from src.environments.wrappers import SimpleObsWrapper, EnrichedObsWrapper, CustomRewardWrapper
from src.environments.rewards import BasicReward


class TestSimpleObsWrapper:
    def test_observation_shape(self):
        env = gymnasium.make("FlappyBird-v0")
        wrapped = SimpleObsWrapper(env)
        obs, _ = wrapped.reset()
        # Simple: (y_bird, velocity, dist_next_pipe, y_gap)
        assert obs.shape == (4,)
        wrapped.close()

    def test_step_returns_correct_shape(self):
        env = gymnasium.make("FlappyBird-v0")
        wrapped = SimpleObsWrapper(env)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(0)
        assert obs.shape == (4,)
        wrapped.close()


class TestEnrichedObsWrapper:
    def test_observation_shape(self):
        env = gymnasium.make("FlappyBird-v0")
        wrapped = EnrichedObsWrapper(env)
        obs, _ = wrapped.reset()
        # Enriched: (y_bird, velocity, dist_next, y_gap, dist_2nd, y_gap_2nd, delta_y)
        assert obs.shape == (7,)
        wrapped.close()


class TestCustomRewardWrapper:
    def test_reward_is_modified(self):
        env = gymnasium.make("FlappyBird-v0")
        wrapped = CustomRewardWrapper(env, reward_fn=BasicReward())
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(0)
        # BasicReward returns 1.0 when alive
        assert reward == 1.0
        wrapped.close()
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_environments/test_wrappers.py -v
```

**Step 3: Implement wrappers**

```python
# src/environments/wrappers.py
import gymnasium
import numpy as np
from src.environments.rewards import RewardFunction


class SimpleObsWrapper(gymnasium.ObservationWrapper):
    """Extracts 4 simple features: y_bird, velocity, dist_next_pipe, y_gap_center."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def observation(self, obs):
        # From flappy-bird-gymnasium default 12-feature obs:
        # obs[9] = player_y, obs[10] = player_velocity
        # obs[2] = next_pipe_x, obs[3] = next_pipe_top_y, obs[5] = next_pipe_bottom_y
        player_y = obs[9]
        velocity = obs[10]
        dist_next = obs[2]
        gap_center = (obs[3] + obs[5]) / 2.0
        return np.array([player_y, velocity, dist_next, gap_center], dtype=np.float32)


class EnrichedObsWrapper(gymnasium.ObservationWrapper):
    """Extracts 7 features: simple + 2nd pipe info + delta to gap."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def observation(self, obs):
        player_y = obs[9]
        velocity = obs[10]
        dist_next = obs[2]
        gap_center = (obs[3] + obs[5]) / 2.0
        dist_2nd = obs[6]
        gap_2nd_center = (obs[7] + obs[8]) / 2.0  # 2nd pipe top/bottom
        delta_y = player_y - gap_center
        return np.array(
            [player_y, velocity, dist_next, gap_center, dist_2nd, gap_2nd_center, delta_y],
            dtype=np.float32,
        )


class CustomRewardWrapper(gymnasium.RewardWrapper):
    """Replaces the default reward with a custom reward function."""

    def __init__(self, env, reward_fn: RewardFunction):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._last_obs = None
        self._last_terminated = False
        self._last_truncated = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._last_terminated = terminated
        self._last_truncated = truncated
        new_reward = self._reward_fn.compute(obs, reward, terminated, truncated)
        return obs, new_reward, terminated, truncated, info
```

**Step 4: Update `src/environments/__init__.py`**

```python
# src/environments/__init__.py
from src.environments.wrappers import SimpleObsWrapper, EnrichedObsWrapper, CustomRewardWrapper
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_environments/ -v
```

Expected: all PASS

**Step 6: Commit**

```bash
git add src/environments/ tests/test_environments/
git commit -m "feat: add observation wrappers and custom reward wrapper"
```

---

### Task 4: BaseAgent interface

**Files:**
- Create: `src/agents/base_agent.py`
- Create: `tests/test_agents/test_base_agent.py`

**Step 1: Write failing test**

```python
# tests/test_agents/test_base_agent.py
import numpy as np
from src.agents.base_agent import BaseAgent


class TestBaseAgent:
    def test_cannot_instantiate_directly(self):
        try:
            agent = BaseAgent(state_dim=4, action_dim=2)
            assert False, "Should not be instantiable"
        except TypeError:
            pass

    def test_subclass_must_implement_methods(self):
        class IncompleteAgent(BaseAgent):
            pass

        try:
            agent = IncompleteAgent(state_dim=4, action_dim=2)
            assert False, "Should not be instantiable"
        except TypeError:
            pass
```

**Step 2: Run test, verify fail**

```bash
python -m pytest tests/test_agents/test_base_agent.py -v
```

**Step 3: Implement BaseAgent**

```python
# src/agents/base_agent.py
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action given a state. If training=True, may explore."""
        pass

    @abstractmethod
    def train_step(self, state, action, reward, next_state, done) -> dict:
        """Perform one training step. Returns dict of metrics (e.g. loss)."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save agent state to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load agent state from disk."""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Return agent info dict for logging (epsilon, lr, etc.)."""
        pass
```

**Step 4: Run tests, verify pass**

```bash
python -m pytest tests/test_agents/test_base_agent.py -v
```

**Step 5: Commit**

```bash
git add src/agents/base_agent.py tests/test_agents/test_base_agent.py
git commit -m "feat: add BaseAgent abstract interface"
```

---

### Task 5: Q-Learning agent

**Files:**
- Create: `src/agents/q_learning.py`
- Create: `tests/test_agents/test_q_learning.py`

**Step 1: Write failing tests**

```python
# tests/test_agents/test_q_learning.py
import numpy as np
import tempfile
from pathlib import Path
from src.agents.q_learning import QLearningAgent


class TestQLearningAgent:
    def test_creation(self):
        agent = QLearningAgent(
            state_dim=4, action_dim=2, n_bins=10,
            lr=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        )
        assert agent.action_dim == 2

    def test_select_action_returns_valid(self):
        agent = QLearningAgent(state_dim=4, action_dim=2, n_bins=10)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        action = agent.select_action(state, training=True)
        assert action in [0, 1]

    def test_greedy_action_no_exploration(self):
        agent = QLearningAgent(state_dim=4, action_dim=2, n_bins=10, epsilon_start=0.0)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        # With epsilon=0, should always pick greedy
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        assert len(set(actions)) == 1  # always same action

    def test_train_step_updates_q_table(self):
        agent = QLearningAgent(state_dim=4, action_dim=2, n_bins=10, lr=1.0, gamma=0.0)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        next_state = np.array([0.5, 0.2, 0.2, 0.5], dtype=np.float32)
        # With lr=1.0 and gamma=0.0, Q(s,a) should become exactly reward
        agent.train_step(state, 0, 5.0, next_state, False)
        key = agent._discretize(state)
        assert agent.q_table[key][0] == 5.0

    def test_save_load(self):
        agent = QLearningAgent(state_dim=4, action_dim=2, n_bins=10)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        agent.train_step(state, 1, 10.0, state, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q_agent"
            agent.save(path)
            agent2 = QLearningAgent(state_dim=4, action_dim=2, n_bins=10)
            agent2.load(path)
            key = agent._discretize(state)
            assert agent2.q_table[key][1] == agent.q_table[key][1]
```

**Step 2: Run tests, verify fail**

```bash
python -m pytest tests/test_agents/test_q_learning.py -v
```

**Step 3: Implement Q-Learning agent**

```python
# src/agents/q_learning.py
import json
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning with state discretization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_bins: int = 10,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(state_dim, action_dim)
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_dim)
        )

    def _discretize(self, state: np.ndarray) -> tuple:
        """Discretize continuous state into bins."""
        clipped = np.clip(state, -1.0, 1.0)
        binned = np.digitize(clipped, np.linspace(-1.0, 1.0, self.n_bins))
        return tuple(binned.tolist())

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        key = self._discretize(state)
        return int(np.argmax(self.q_table[key]))

    def train_step(self, state, action, reward, next_state, done) -> dict:
        key = self._discretize(state)
        next_key = self._discretize(next_state)
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return {"q_value": float(self.q_table[key][action])}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "q_table.pkl", "wb") as f:
            pickle.dump(dict(self.q_table), f)
        with open(path / "params.json", "w") as f:
            json.dump({"epsilon": self.epsilon}, f)

    def load(self, path: Path) -> None:
        with open(path / "q_table.pkl", "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim), data)
        with open(path / "params.json", "r") as f:
            params = json.load(f)
            self.epsilon = params["epsilon"]

    def get_info(self) -> dict:
        return {"epsilon": self.epsilon, "q_table_size": len(self.q_table)}
```

**Step 4: Run tests, verify pass**

```bash
python -m pytest tests/test_agents/test_q_learning.py -v
```

**Step 5: Commit**

```bash
git add src/agents/q_learning.py tests/test_agents/test_q_learning.py
git commit -m "feat: add Q-Learning tabular agent"
```

---

### Task 6: DQN agent

**Files:**
- Create: `src/agents/dqn.py`
- Create: `tests/test_agents/test_dqn.py`

**Step 1: Write failing tests**

```python
# tests/test_agents/test_dqn.py
import numpy as np
import tempfile
from pathlib import Path
import torch
from src.agents.dqn import DQNAgent


class TestDQNAgent:
    def test_creation(self):
        agent = DQNAgent(state_dim=4, action_dim=2, hidden_dims=[64, 64])
        assert agent.action_dim == 2

    def test_select_action(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        action = agent.select_action(state, training=True)
        assert action in [0, 1]

    def test_train_step_returns_loss(self):
        agent = DQNAgent(state_dim=4, action_dim=2, batch_size=4, buffer_size=100)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        # Fill buffer enough for one batch
        for _ in range(10):
            next_state = state + np.random.randn(4).astype(np.float32) * 0.1
            agent.train_step(state, 0, 1.0, next_state, False)
            state = next_state
        metrics = agent.train_step(state, 1, -1.0, state, True)
        assert "loss" in metrics

    def test_save_load(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        q_before = agent._get_q_values(state)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dqn"
            agent.save(path)
            agent2 = DQNAgent(state_dim=4, action_dim=2)
            agent2.load(path)
            q_after = agent2._get_q_values(state)
            np.testing.assert_array_almost_equal(q_before, q_after)

    def test_target_network_updates(self):
        agent = DQNAgent(state_dim=4, action_dim=2, tau=1.0)
        # With tau=1.0, hard copy
        agent._update_target()
        for p, tp in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.equal(p.data, tp.data)
```

**Step 2: Run tests, verify fail**

```bash
python -m pytest tests/test_agents/test_dqn.py -v
```

**Step 3: Implement DQN agent**

```python
# src/agents/dqn.py
import json
import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.agents.base_agent import BaseAgent


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Deep Q-Network with experience replay and target network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        tau: float = 0.005,
        train_every: int = 4,
    ):
        super().__init__(state_dim, action_dim)
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.train_every = train_every
        self.step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(t).cpu().numpy()[0]

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return int(np.argmax(self._get_q_values(state)))

    def _update_target(self):
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train_step(self, state, action, reward, next_state, done) -> dict:
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.buffer) < self.batch_size:
            return {}

        if self.step_count % self.train_every != 0:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._update_target()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "q_mean": q_values.mean().item()}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path / "q_net.pt")
        torch.save(self.target_net.state_dict(), path / "target_net.pt")
        with open(path / "params.json", "w") as f:
            json.dump({"epsilon": self.epsilon, "step_count": self.step_count}, f)

    def load(self, path: Path) -> None:
        self.q_net.load_state_dict(torch.load(path / "q_net.pt", weights_only=True))
        self.target_net.load_state_dict(torch.load(path / "target_net.pt", weights_only=True))
        with open(path / "params.json", "r") as f:
            params = json.load(f)
            self.epsilon = params["epsilon"]
            self.step_count = params["step_count"]

    def get_info(self) -> dict:
        return {"epsilon": self.epsilon, "step_count": self.step_count, "buffer_size": len(self.buffer)}
```

**Step 4: Run tests, verify pass**

```bash
python -m pytest tests/test_agents/test_dqn.py -v
```

**Step 5: Commit**

```bash
git add src/agents/dqn.py tests/test_agents/test_dqn.py
git commit -m "feat: add DQN agent with replay buffer and target network"
```

---

### Task 7: Double DQN agent

**Files:**
- Create: `src/agents/double_dqn.py`
- Create: `tests/test_agents/test_double_dqn.py`

**Step 1: Write failing tests**

```python
# tests/test_agents/test_double_dqn.py
import numpy as np
import torch
from src.agents.double_dqn import DoubleDQNAgent


class TestDoubleDQNAgent:
    def test_creation(self):
        agent = DoubleDQNAgent(state_dim=4, action_dim=2)
        assert agent.action_dim == 2

    def test_select_action(self):
        agent = DoubleDQNAgent(state_dim=4, action_dim=2)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        action = agent.select_action(state)
        assert action in [0, 1]

    def test_uses_double_q_learning(self):
        """Double DQN uses online net to select action, target net to evaluate."""
        agent = DoubleDQNAgent(state_dim=4, action_dim=2, batch_size=4, buffer_size=100)
        state = np.array([0.5, 0.1, 0.3, 0.5], dtype=np.float32)
        for _ in range(10):
            ns = state + np.random.randn(4).astype(np.float32) * 0.1
            agent.train_step(state, 0, 1.0, ns, False)
            state = ns
        metrics = agent.train_step(state, 1, -1.0, state, True)
        assert "loss" in metrics
```

**Step 2: Run tests, verify fail**

```bash
python -m pytest tests/test_agents/test_double_dqn.py -v
```

**Step 3: Implement Double DQN (inherits from DQN)**

```python
# src/agents/double_dqn.py
import torch
import torch.nn as nn
from src.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN: uses online network for action selection, target for evaluation."""

    def train_step(self, state, action, reward, next_state, done) -> dict:
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.buffer) < self.batch_size:
            return {}

        if self.step_count % self.train_every != 0:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select actions with online net, evaluate with target net
            best_actions = self.q_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._update_target()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "q_mean": q_values.mean().item()}
```

**Step 4: Update `src/agents/__init__.py`**

```python
# src/agents/__init__.py
from src.agents.base_agent import BaseAgent
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent
```

**Step 5: Run all agent tests**

```bash
python -m pytest tests/test_agents/ -v
```

**Step 6: Commit**

```bash
git add src/agents/ tests/test_agents/
git commit -m "feat: add Double DQN agent"
```

---

### Task 8: Training config system

**Files:**
- Create: `src/training/config.py`
- Create: `configs/default.yaml`
- Create: `tests/test_training/test_config.py`

**Step 1: Write failing tests**

```python
# tests/test_training/test_config.py
import tempfile
from pathlib import Path
from src.training.config import ExperimentConfig, load_config


class TestConfig:
    def test_load_yaml(self):
        yaml_content = """
agent: dqn
observation: simple
reward: basic
hyperparams:
  lr: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  hidden_dims: [128, 128]
  batch_size: 64
  buffer_size: 50000
training:
  episodes: 1000
  max_steps: 500
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(Path(f.name))

        assert config.agent == "dqn"
        assert config.observation == "simple"
        assert config.reward == "basic"
        assert config.hyperparams["lr"] == 0.001
        assert config.training["episodes"] == 1000

    def test_config_creates_agent_name(self):
        config = ExperimentConfig(
            agent="dqn", observation="simple", reward="basic",
            hyperparams={"lr": 0.001}, training={"episodes": 100},
        )
        assert "dqn" in config.experiment_name
        assert "simple" in config.experiment_name
```

**Step 2: Run tests, verify fail**

```bash
python -m pytest tests/test_training/test_config.py -v
```

**Step 3: Implement config**

```python
# src/training/config.py
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    agent: str
    observation: str
    reward: str
    hyperparams: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)

    @property
    def experiment_name(self) -> str:
        return f"{self.agent}_{self.observation}_{self.reward}"


def load_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
```

**Step 4: Create default config**

```yaml
# configs/default.yaml
agent: dqn
observation: simple
reward: basic
hyperparams:
  lr: 0.0005
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  hidden_dims: [128, 128]
  batch_size: 64
  buffer_size: 50000
  tau: 0.005
training:
  episodes: 1000
  max_steps: 500
  save_every: 100
  log_every: 10
```

**Step 5: Run tests, verify pass**

```bash
python -m pytest tests/test_training/test_config.py -v
```

**Step 6: Commit**

```bash
git add src/training/config.py configs/default.yaml tests/test_training/test_config.py
git commit -m "feat: add experiment config system with YAML loading"
```

---

### Task 9: Metrics logger

**Files:**
- Create: `src/training/logger.py`
- Create: `tests/test_training/test_logger.py`

**Step 1: Write failing tests**

```python
# tests/test_training/test_logger.py
from src.training.logger import MetricsLogger


class TestMetricsLogger:
    def test_log_episode(self):
        logger = MetricsLogger()
        logger.log_episode(episode=1, score=5, reward=100.0, steps=200, agent_info={"epsilon": 0.9})
        assert len(logger.history) == 1
        assert logger.history[0]["score"] == 5

    def test_get_recent(self):
        logger = MetricsLogger()
        for i in range(20):
            logger.log_episode(episode=i, score=i, reward=float(i), steps=100)
        recent = logger.get_recent(n=10)
        assert len(recent) == 10
        assert recent[-1]["score"] == 19

    def test_best_score(self):
        logger = MetricsLogger()
        logger.log_episode(episode=0, score=3, reward=10.0, steps=50)
        logger.log_episode(episode=1, score=7, reward=20.0, steps=80)
        logger.log_episode(episode=2, score=2, reward=5.0, steps=30)
        assert logger.best_score == 7

    def test_average_score(self):
        logger = MetricsLogger()
        for i in range(10):
            logger.log_episode(episode=i, score=10, reward=100.0, steps=100)
        assert logger.average_score(n=10) == 10.0
```

**Step 2: Run tests, verify fail**

```bash
python -m pytest tests/test_training/test_logger.py -v
```

**Step 3: Implement logger**

```python
# src/training/logger.py
from dataclasses import dataclass, field


class MetricsLogger:
    def __init__(self):
        self.history: list[dict] = []

    def log_episode(
        self,
        episode: int,
        score: int,
        reward: float,
        steps: int,
        agent_info: dict | None = None,
    ):
        entry = {
            "episode": episode,
            "score": score,
            "total_reward": reward,
            "steps": steps,
        }
        if agent_info:
            entry.update(agent_info)
        self.history.append(entry)

    def get_recent(self, n: int = 10) -> list[dict]:
        return self.history[-n:]

    @property
    def best_score(self) -> int:
        if not self.history:
            return 0
        return max(e["score"] for e in self.history)

    def average_score(self, n: int = 100) -> float:
        recent = self.history[-n:]
        if not recent:
            return 0.0
        return sum(e["score"] for e in recent) / len(recent)
```

**Step 4: Run tests, verify pass**

```bash
python -m pytest tests/test_training/test_logger.py -v
```

**Step 5: Commit**

```bash
git add src/training/logger.py tests/test_training/test_logger.py
git commit -m "feat: add metrics logger for episode tracking"
```

---

### Task 10: Trainer (training loop)

**Files:**
- Create: `src/training/trainer.py`
- Create: `tests/test_training/test_trainer.py`

**Step 1: Write failing tests**

```python
# tests/test_training/test_trainer.py
import gymnasium
import numpy as np
import flappy_bird_gymnasium
from src.training.trainer import Trainer
from src.training.config import ExperimentConfig
from src.training.logger import MetricsLogger
from src.agents.q_learning import QLearningAgent
from src.environments.wrappers import SimpleObsWrapper
from src.environments.rewards import BasicReward


class TestTrainer:
    def test_build_env(self):
        config = ExperimentConfig(
            agent="q_learning", observation="simple", reward="basic",
            hyperparams={}, training={"episodes": 2, "max_steps": 50},
        )
        trainer = Trainer(config)
        env = trainer._build_env()
        obs, _ = env.reset()
        assert obs.shape == (4,)
        env.close()

    def test_build_agent(self):
        config = ExperimentConfig(
            agent="q_learning", observation="simple", reward="basic",
            hyperparams={"n_bins": 10}, training={"episodes": 2, "max_steps": 50},
        )
        trainer = Trainer(config)
        env = trainer._build_env()
        agent = trainer._build_agent(env)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        env.close()

    def test_train_runs(self):
        config = ExperimentConfig(
            agent="q_learning", observation="simple", reward="basic",
            hyperparams={"n_bins": 10, "epsilon_start": 0.5},
            training={"episodes": 3, "max_steps": 50},
        )
        trainer = Trainer(config)
        logger = trainer.train()
        assert len(logger.history) == 3
```

**Step 2: Run tests, verify fail**

```bash
python -m pytest tests/test_training/test_trainer.py -v
```

**Step 3: Implement trainer**

```python
# src/training/trainer.py
from pathlib import Path
import gymnasium
import flappy_bird_gymnasium
from src.agents.base_agent import BaseAgent
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent
from src.environments.wrappers import SimpleObsWrapper, EnrichedObsWrapper, CustomRewardWrapper
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward
from src.training.config import ExperimentConfig
from src.training.logger import MetricsLogger

AGENT_MAP = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
}

OBS_WRAPPER_MAP = {
    "simple": SimpleObsWrapper,
    "enriched": EnrichedObsWrapper,
    "raw": None,
}

REWARD_MAP = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
}


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _build_env(self) -> gymnasium.Env:
        env = gymnasium.make("FlappyBird-v0")
        reward_cls = REWARD_MAP.get(self.config.reward)
        if reward_cls:
            env = CustomRewardWrapper(env, reward_fn=reward_cls())
        obs_wrapper = OBS_WRAPPER_MAP.get(self.config.observation)
        if obs_wrapper:
            env = obs_wrapper(env)
        return env

    def _build_agent(self, env: gymnasium.Env) -> BaseAgent:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent_cls = AGENT_MAP[self.config.agent]
        return agent_cls(state_dim=state_dim, action_dim=action_dim, **self.config.hyperparams)

    def train(self, render: bool = False) -> MetricsLogger:
        env = self._build_env()
        agent = self._build_agent(env)
        logger = MetricsLogger()

        episodes = self.config.training.get("episodes", 1000)
        max_steps = self.config.training.get("max_steps", 500)
        save_every = self.config.training.get("save_every", 100)

        for ep in range(episodes):
            state, _ = env.reset()
            total_reward = 0.0
            score = 0

            for step in range(max_steps):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.train_step(state, action, reward, next_state, terminated or truncated)
                total_reward += reward
                score = info.get("score", score)
                state = next_state
                if terminated or truncated:
                    break

            logger.log_episode(
                episode=ep, score=score, reward=total_reward,
                steps=step + 1, agent_info=agent.get_info(),
            )

            if save_every and (ep + 1) % save_every == 0:
                save_path = Path("models") / self.config.experiment_name / f"ep_{ep+1}"
                agent.save(save_path)

        env.close()
        return agent, logger
```

**Step 4: Update `src/training/__init__.py`**

```python
# src/training/__init__.py
from src.training.config import ExperimentConfig, load_config
from src.training.trainer import Trainer
from src.training.logger import MetricsLogger
```

**Step 5: Run tests, verify pass**

```bash
python -m pytest tests/test_training/ -v
```

**Step 6: Commit**

```bash
git add src/training/ tests/test_training/
git commit -m "feat: add training pipeline with env/agent builders"
```

---

### Task 11: Pygame renderer

**Files:**
- Create: `src/visualization/renderer.py`

**Step 1: Implement renderer**

Note: Pygame rendering is hard to unit test. We test manually.

```python
# src/visualization/renderer.py
import gymnasium
import flappy_bird_gymnasium
import pygame
from src.agents.base_agent import BaseAgent
from src.environments.wrappers import SimpleObsWrapper, EnrichedObsWrapper, CustomRewardWrapper
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward
from src.visualization.overlay import Overlay


OBS_WRAPPER_MAP = {
    "simple": SimpleObsWrapper,
    "enriched": EnrichedObsWrapper,
    "raw": None,
}

REWARD_MAP = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
}


class GameRenderer:
    """Renders Flappy Bird with an RL agent playing, with metrics overlay."""

    def __init__(self, agent: BaseAgent, obs_type: str = "simple", reward_type: str = "basic", fps: int = 30):
        self.agent = agent
        self.obs_type = obs_type
        self.reward_type = reward_type
        self.fps = fps
        self.overlay = Overlay()

    def _build_env(self) -> gymnasium.Env:
        env = gymnasium.make("FlappyBird-v0", render_mode="human")
        reward_cls = REWARD_MAP.get(self.reward_type)
        if reward_cls:
            env = CustomRewardWrapper(env, reward_fn=reward_cls())
        obs_wrapper = OBS_WRAPPER_MAP.get(self.obs_type)
        if obs_wrapper:
            env = obs_wrapper(env)
        return env

    def run(self, num_episodes: int = 10):
        """Run the agent playing Flappy Bird with visual rendering."""
        env = self._build_env()
        clock = pygame.time.Clock()

        for ep in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            score = 0
            step = 0
            done = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        env.close()
                        return

                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                score = info.get("score", score)
                step += 1
                done = terminated or truncated

                self.overlay.update(
                    episode=ep + 1,
                    total_episodes=num_episodes,
                    score=score,
                    total_reward=total_reward,
                    steps=step,
                    agent_info=self.agent.get_info(),
                    agent_name=type(self.agent).__name__,
                )
                self.overlay.draw(pygame.display.get_surface())
                pygame.display.flip()
                clock.tick(self.fps)

        env.close()
```

**Step 2: Commit**

```bash
git add src/visualization/renderer.py
git commit -m "feat: add Pygame game renderer with agent replay"
```

---

### Task 12: Metrics overlay

**Files:**
- Create: `src/visualization/overlay.py`

**Step 1: Implement overlay**

```python
# src/visualization/overlay.py
import pygame


class Overlay:
    """Draws metrics overlay on top of the Pygame surface."""

    def __init__(self):
        self.data = {}
        self._font = None
        self._bg_color = (0, 0, 0, 160)
        self._text_color = (255, 255, 255)
        self._highlight_color = (0, 255, 100)

    def _get_font(self, size: int = 18) -> pygame.font.Font:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", size, bold=True)
        return self._font

    def update(
        self,
        episode: int,
        total_episodes: int,
        score: int,
        total_reward: float,
        steps: int,
        agent_info: dict,
        agent_name: str,
    ):
        self.data = {
            "agent": agent_name,
            "episode": f"{episode}/{total_episodes}",
            "score": score,
            "reward": f"{total_reward:.1f}",
            "steps": steps,
            **{k: (f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in agent_info.items()},
        }

    def draw(self, surface: pygame.Surface):
        if not self.data:
            return

        font = self._get_font()
        padding = 8
        line_height = 22
        width = 280
        height = padding * 2 + line_height * len(self.data)

        # Semi-transparent background
        overlay_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay_surface.fill(self._bg_color)
        surface.blit(overlay_surface, (5, 5))

        y = 5 + padding
        for key, value in self.data.items():
            label = font.render(f"{key}: ", True, self._text_color)
            val = font.render(str(value), True, self._highlight_color)
            surface.blit(label, (10, y))
            surface.blit(val, (10 + label.get_width(), y))
            y += line_height
```

**Step 2: Update `src/visualization/__init__.py`**

```python
# src/visualization/__init__.py
from src.visualization.renderer import GameRenderer
from src.visualization.overlay import Overlay
```

**Step 3: Commit**

```bash
git add src/visualization/
git commit -m "feat: add metrics overlay for Pygame visualization"
```

---

### Task 13: Main entry point

**Files:**
- Create: `main.py`

**Step 1: Implement main**

```python
# main.py
"""RL Flappy Bird Demo — Train and visualize RL agents playing Flappy Bird."""
import argparse
import sys
from pathlib import Path
from src.training.config import load_config, ExperimentConfig
from src.training.trainer import Trainer
from src.visualization.renderer import GameRenderer
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent

AGENT_MAP = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
}


def cmd_train(args):
    config = load_config(Path(args.config))
    print(f"Training: {config.experiment_name}")
    trainer = Trainer(config)
    agent, logger = trainer.train()
    print(f"Training complete. Best score: {logger.best_score}")
    print(f"Average score (last 100): {logger.average_score(100):.1f}")

    if args.save:
        save_path = Path("models") / config.experiment_name / "final"
        agent.save(save_path)
        print(f"Model saved to {save_path}")


def cmd_play(args):
    config = load_config(Path(args.config))
    agent_cls = AGENT_MAP[config.agent]
    state_dim = 4 if config.observation == "simple" else 7 if config.observation == "enriched" else 12
    agent = agent_cls(state_dim=state_dim, action_dim=2, **config.hyperparams)

    model_path = Path(args.model)
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    renderer = GameRenderer(
        agent=agent,
        obs_type=config.observation,
        reward_type=config.reward,
        fps=args.fps,
    )
    renderer.run(num_episodes=args.episodes)


def main():
    parser = argparse.ArgumentParser(description="RL Flappy Bird Demo")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument("--config", required=True, help="Path to YAML config")
    train_parser.add_argument("--save", action="store_true", help="Save model after training")

    play_parser = subparsers.add_parser("play", help="Watch a trained agent play")
    play_parser.add_argument("--config", required=True, help="Path to YAML config")
    play_parser.add_argument("--model", required=True, help="Path to saved model directory")
    play_parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    play_parser.add_argument("--fps", type=int, default=30, help="Frames per second")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "play":
        cmd_play(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add main entry point with train/play CLI commands"
```

---

### Task 14: Integration — create experiment configs and test end-to-end

**Files:**
- Create: `configs/q_learning_simple_basic.yaml`
- Create: `configs/dqn_simple_basic.yaml`
- Create: `configs/dqn_enriched_centered.yaml`
- Create: `configs/double_dqn_simple_distance.yaml`

**Step 1: Create config variants**

```yaml
# configs/q_learning_simple_basic.yaml
agent: q_learning
observation: simple
reward: basic
hyperparams:
  n_bins: 10
  lr: 0.1
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
training:
  episodes: 2000
  max_steps: 500
  save_every: 500
```

```yaml
# configs/dqn_simple_basic.yaml
agent: dqn
observation: simple
reward: basic
hyperparams:
  lr: 0.0005
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  hidden_dims: [128, 128]
  batch_size: 64
  buffer_size: 50000
  tau: 0.005
training:
  episodes: 1000
  max_steps: 500
  save_every: 200
```

```yaml
# configs/dqn_enriched_centered.yaml
agent: dqn
observation: enriched
reward: centered
hyperparams:
  lr: 0.0005
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  hidden_dims: [128, 128]
  batch_size: 64
  buffer_size: 50000
  tau: 0.005
training:
  episodes: 1000
  max_steps: 500
  save_every: 200
```

```yaml
# configs/double_dqn_simple_distance.yaml
agent: double_dqn
observation: simple
reward: distance
hyperparams:
  lr: 0.0005
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  hidden_dims: [128, 128]
  batch_size: 64
  buffer_size: 50000
  tau: 0.005
training:
  episodes: 1000
  max_steps: 500
  save_every: 200
```

**Step 2: Run quick smoke test**

```bash
python main.py train --config configs/q_learning_simple_basic.yaml --save
```

Wait for a few episodes to confirm it runs without errors (Ctrl+C to stop early).

**Step 3: Commit**

```bash
git add configs/ main.py
git commit -m "feat: add experiment configs for all agent/observation/reward combos"
```

---

## Usage

**Train an agent:**
```bash
python main.py train --config configs/dqn_simple_basic.yaml --save
```

**Watch it play:**
```bash
python main.py play --config configs/dqn_simple_basic.yaml --model models/dqn_simple_basic/final --episodes 5
```
