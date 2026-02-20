"""Training loop for RL Flappy Bird experiments.

Builds the environment (with reward and observation wrappers) and agent
from an ExperimentConfig, then runs the training loop and returns the
trained agent together with the metrics logger.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import flappy_bird_gymnasium  # noqa: F401 — registers FlappyBird-v0
import gymnasium
import numpy as np

from src.agents import QLearningAgent, DQNAgent, DoubleDQNAgent, BaseAgent
from src.environments.wrappers import SimpleObsWrapper, EnrichedObsWrapper, CustomRewardWrapper
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward
from src.training.config import ExperimentConfig
from src.training.logger import MetricsLogger


# ---------------------------------------------------------------------------
# Lookup maps
# ---------------------------------------------------------------------------

AGENT_MAP: dict[str, type] = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
}

OBS_WRAPPER_MAP: dict[str, type | None] = {
    "simple": SimpleObsWrapper,
    "enriched": EnrichedObsWrapper,
    "raw": None,
}

REWARD_MAP: dict[str, type] = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
}


class Trainer:
    """Orchestrates environment construction, agent creation, and training.

    Args:
        config: An ExperimentConfig describing the full experiment.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Environment construction
    # ------------------------------------------------------------------

    def _build_env(self) -> gymnasium.Env:
        """Build a Gymnasium environment with the configured wrappers.

        Wrapping order:
            1. Base FlappyBird-v0 environment (use_lidar=False)
            2. CustomRewardWrapper with the selected reward function
            3. Observation wrapper (simple / enriched), unless "raw"

        Returns:
            The fully wrapped Gymnasium environment.
        """
        env = gymnasium.make("FlappyBird-v0", use_lidar=False)

        # Apply reward wrapper
        reward_cls = REWARD_MAP[self.config.reward]
        env = CustomRewardWrapper(env, reward_fn=reward_cls())

        # Apply observation wrapper (None for "raw")
        obs_wrapper_cls = OBS_WRAPPER_MAP[self.config.observation]
        if obs_wrapper_cls is not None:
            env = obs_wrapper_cls(env)

        return env

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _build_agent(self, env: gymnasium.Env) -> BaseAgent:
        """Instantiate the configured agent with dimensions from the env.

        Only hyperparameters whose names match the agent constructor's
        signature are forwarded; extras are silently ignored so that a
        single config can be shared across different agent types.

        Args:
            env: The (wrapped) environment to derive state/action dims from.

        Returns:
            A BaseAgent subclass instance.
        """
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent_cls = AGENT_MAP[self.config.agent]
        hyperparams = dict(self.config.hyperparams)

        # Filter to only params accepted by this agent's __init__
        sig = inspect.signature(agent_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in hyperparams.items() if k in valid_params}

        return agent_cls(state_dim=state_dim, action_dim=action_dim, **filtered)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> tuple[BaseAgent, MetricsLogger]:
        """Run the full training loop.

        For each episode:
            1. Reset the environment.
            2. Interact for up to max_steps (or until termination/truncation).
            3. Log episode metrics.
            4. Periodically save the agent.

        Returns:
            A tuple of (trained agent, metrics logger).
        """
        env = self._build_env()
        agent = self._build_agent(env)
        logger = MetricsLogger()

        episodes = self.config.training.get("episodes", 1000)
        max_steps = self.config.training.get("max_steps", 500)
        save_every = self.config.training.get("save_every", 100)
        log_every = self.config.training.get("log_every", 10)

        print(f"Training {self.config.experiment_name} for {episodes} episodes...")

        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_score = 0
            steps = 0

            for step in range(max_steps):
                action = agent.select_action(np.asarray(obs, dtype=np.float32), training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)

                agent.train_step(
                    np.asarray(obs, dtype=np.float32),
                    action,
                    reward,
                    np.asarray(next_obs, dtype=np.float32),
                    terminated or truncated,
                )

                episode_reward += reward
                episode_score = info.get("score", episode_score)
                obs = next_obs
                steps += 1

                if terminated or truncated:
                    break

            agent_info = agent.get_info()
            logger.log_episode(
                episode=episode,
                score=episode_score,
                reward=episode_reward,
                steps=steps,
                agent_info=agent_info,
            )

            # Periodic logging
            if (episode + 1) % log_every == 0:
                avg = logger.average_score(n=log_every)
                eps = agent_info.get("epsilon", "N/A")
                print(f"  Episode {episode + 1}/{episodes} | "
                      f"Avg score: {avg:.1f} | Best: {logger.best_score} | "
                      f"Epsilon: {eps}")

            # Periodic save
            if (episode + 1) % save_every == 0:
                save_dir = Path("models") / self.config.experiment_name / f"ep_{episode + 1}"
                agent.save(save_dir)

        env.close()
        return agent, logger
