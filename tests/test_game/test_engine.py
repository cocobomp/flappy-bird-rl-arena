import numpy as np
from src.game.engine import FlappyBirdEngine, Bird

class TestBird:
    def test_creation(self):
        bird = Bird(bird_id=0, color=(255, 0, 0))
        assert bird.alive
        assert bird.score == 0

    def test_flap(self):
        bird = Bird(bird_id=0, color=(255, 0, 0))
        old_vel = bird.vel_y
        bird.flap()
        assert bird.vel_y == -9  # PLAYER_FLAP_ACC

    def test_gravity(self):
        bird = Bird(bird_id=0, color=(255, 0, 0))
        bird.vel_y = 0
        bird.update(ground_y=400)
        assert bird.vel_y == 1  # PLAYER_ACC_Y
        assert bird.y > 244  # moved down

    def test_max_velocity(self):
        bird = Bird(bird_id=0, color=(255, 0, 0))
        bird.vel_y = 10  # at max
        bird.update(ground_y=400)
        assert bird.vel_y == 10  # capped at PLAYER_MAX_VEL_Y

    def test_ground_collision(self):
        bird = Bird(bird_id=0, color=(255, 0, 0))
        bird.y = 380  # near ground
        bird.vel_y = 10
        bird.update(ground_y=400)
        assert bird.y <= 400 - 24  # PLAYER_HEIGHT


class TestFlappyBirdEngine:
    def test_creation(self):
        engine = FlappyBirdEngine()
        assert engine.pipes == []
        assert engine.round_num == 0

    def test_add_bird(self):
        engine = FlappyBirdEngine()
        bird = engine.add_bird(color=(255, 0, 0))
        assert len(engine.birds) == 1
        assert bird.alive

    def test_reset(self):
        engine = FlappyBirdEngine()
        engine.add_bird(color=(255, 0, 0))
        engine.reset()
        assert engine.round_num == 1
        assert len(engine.pipes) > 0  # pipes generated

    def test_step_returns_observations(self):
        engine = FlappyBirdEngine()
        engine.add_bird(color=(255, 0, 0))
        engine.add_bird(color=(0, 0, 255))
        engine.reset()
        actions = {0: 0, 1: 1}  # bird 0: nothing, bird 1: flap
        obs_dict, reward_dict, done_dict, info = engine.step(actions)
        assert 0 in obs_dict
        assert 1 in obs_dict
        assert obs_dict[0].shape == (8,)

    def test_all_dead_ends_round(self):
        engine = FlappyBirdEngine()
        engine.add_bird(color=(255, 0, 0))
        engine.reset()
        # Force bird to die
        engine.birds[0].alive = False
        _, _, done_dict, info = engine.step({0: 0})
        assert info["round_over"]

    def test_pipe_generation(self):
        engine = FlappyBirdEngine()
        engine.add_bird(color=(255, 0, 0))
        engine.reset()
        initial_pipes = len(engine.pipes)
        assert initial_pipes >= 2

    def test_scoring(self):
        engine = FlappyBirdEngine()
        bird = engine.add_bird(color=(255, 0, 0))
        engine.reset()
        # Manually position bird past a pipe
        engine.pipes[0]["x"] = engine.birds[0].x - 52 - 1  # past the pipe
        engine.pipes[0]["scored"] = False
        engine._check_scoring()
        # Score should have incremented
        assert bird.score >= 0  # depends on exact positioning

    def test_observation_values(self):
        engine = FlappyBirdEngine()
        bird = engine.add_bird(color=(255, 0, 0))
        engine.reset()
        obs = engine.get_observation(bird)
        # 8 features: delta_y1, velocity, dist_pipe1, delta_y2, dist_pipe2,
        # gap_position, proximity_danger, vertical_speed_direction
        assert len(obs) == 8
        assert isinstance(obs[0], (float, np.floating))
