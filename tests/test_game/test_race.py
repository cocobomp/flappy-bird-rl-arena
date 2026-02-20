import numpy as np
from src.game.race import RaceManager, BirdEntry


class TestBirdEntry:
    def test_creation(self):
        entry = BirdEntry(algo="dqn", reward="basic", strategy_name="guided",
                          color=(255, 0, 0), state_dim=5)
        assert entry.agent is not None
        assert entry.reward_fn is not None
        assert entry.strategy is not None
        assert entry.best_score == 0


class TestRaceManager:
    def test_add_bird(self):
        rm = RaceManager(render=False)
        rm.add_bird(algo="dqn", reward="basic")
        assert len(rm.entries) == 1

    def test_add_multiple_birds(self):
        rm = RaceManager(render=False)
        rm.add_bird(algo="dqn", reward="basic")
        rm.add_bird(algo="q_learning", reward="centered")
        rm.add_bird(algo="double_dqn", reward="distance")
        assert len(rm.entries) == 3

    def test_run_one_round(self):
        rm = RaceManager(render=False)
        rm.add_bird(algo="dqn", reward="basic")
        rm.reset_round()
        # Run until round over
        for _ in range(1000):
            done = rm.step_frame()
            if done:
                break
        assert rm.engine.round_num >= 1
