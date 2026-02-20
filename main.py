"""RL Flappy Bird Demo -- Train and visualize RL agents playing Flappy Bird."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agents import DQNAgent, DoubleDQNAgent, QLearningAgent
from src.training.config import load_config
from src.training.trainer import Trainer
from src.visualization.renderer import GameRenderer

AGENT_MAP: dict[str, type] = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
}

# State dimensions for each observation type
STATE_DIM_MAP: dict[str, int] = {
    "simple": 4,
    "enriched": 7,
    "raw": 12,
}

# Flappy Bird action space size (flap or no-flap)
ACTION_DIM = 2


def cmd_train(args: argparse.Namespace) -> None:
    """Train an agent using the configuration file.

    Loads the YAML config, creates a Trainer, runs training, prints
    summary results, and optionally saves the trained model.

    Args:
        args: Parsed CLI arguments with ``config`` path and ``save`` flag.
    """
    config = load_config(args.config)
    print(f"Training experiment: {config.experiment_name}")
    print(f"  Agent: {config.agent}")
    print(f"  Observation: {config.observation}")
    print(f"  Reward: {config.reward}")
    print(f"  Episodes: {config.training.get('episodes', 'N/A')}")

    trainer = Trainer(config)
    agent, logger = trainer.train()

    # Print summary
    print("\nTraining complete.")
    print(f"  Agent info: {agent.get_info()}")

    if args.save:
        save_dir = Path("models") / config.experiment_name
        agent.save(save_dir)
        print(f"  Model saved to: {save_dir}")


def cmd_play(args: argparse.Namespace) -> None:
    """Load a trained agent and visualise it playing Flappy Bird.

    Reads the config to determine the agent type, observation wrapper,
    and reward function. Creates the agent, loads saved weights, and
    launches the GameRenderer.

    Args:
        args: Parsed CLI arguments with ``config``, ``model``, ``episodes``,
              and ``fps``.
    """
    config = load_config(args.config)

    # Determine state dimension from observation type
    state_dim = STATE_DIM_MAP.get(config.observation)
    if state_dim is None:
        print(
            f"Error: unknown observation type '{config.observation}'. "
            f"Expected one of: {list(STATE_DIM_MAP.keys())}"
        )
        sys.exit(1)

    # Create agent
    agent_cls = AGENT_MAP.get(config.agent)
    if agent_cls is None:
        print(
            f"Error: unknown agent type '{config.agent}'. "
            f"Expected one of: {list(AGENT_MAP.keys())}"
        )
        sys.exit(1)

    agent = agent_cls(
        state_dim=state_dim,
        action_dim=ACTION_DIM,
        **config.hyperparams,
    )

    # Load saved model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model path not found: {model_path}")
        sys.exit(1)

    agent.load(model_path)
    print(f"Loaded model from: {model_path}")
    print(f"  Agent: {config.agent} ({type(agent).__name__})")
    print(f"  Observation: {config.observation} (state_dim={state_dim})")
    print(f"  Reward: {config.reward}")
    print(f"  Episodes: {args.episodes}, FPS: {args.fps}")

    renderer = GameRenderer(
        agent=agent,
        obs_type=config.observation,
        reward_type=config.reward,
        fps=args.fps,
    )
    renderer.run(num_episodes=args.episodes)


def main() -> None:
    """Parse CLI arguments and dispatch to train or play subcommand."""
    parser = argparse.ArgumentParser(
        description="RL Flappy Bird Demo -- Train and visualize RL agents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- train subcommand ---
    train_parser = subparsers.add_parser(
        "train",
        help="Train an RL agent on Flappy Bird.",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment configuration file.",
    )
    train_parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the trained model to models/<experiment_name>/.",
    )

    # --- play subcommand ---
    play_parser = subparsers.add_parser(
        "play",
        help="Visualize a trained agent playing Flappy Bird.",
    )
    play_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment configuration file.",
    )
    play_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved model directory.",
    )
    play_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5).",
    )
    play_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for rendering (default: 30).",
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "play":
        cmd_play(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
