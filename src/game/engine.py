"""Multi-bird Flappy Bird game engine.

Manages N birds simultaneously in a shared pipe environment.
Uses exact physics from flappy-bird-gymnasium.
"""
import numpy as np

# Physics constants (from flappy-bird-gymnasium)
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24
PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP = 100  # vertical gap between upper and lower pipe
PIPE_VEL_X = -4  # horizontal pipe speed (leftward)
PLAYER_ACC_Y = 1  # gravity
PLAYER_FLAP_ACC = -9  # upward velocity on flap
PLAYER_MAX_VEL_Y = 10  # max downward velocity
PLAYER_MIN_VEL_Y = -8  # max upward velocity
GROUND_Y = int(SCREEN_HEIGHT * 0.79)  # ~404
PLAYER_START_X = int(SCREEN_WIDTH * 0.2)  # ~57
PLAYER_START_Y = int((SCREEN_HEIGHT - PLAYER_HEIGHT) / 2)  # ~244
GAP_YS = [20, 30, 40, 50, 60, 70, 80, 90]  # possible gap positions
GAP_OFFSET = int(GROUND_Y * 0.2)  # ~80


class Bird:
    """A single bird with its own physics state."""

    def __init__(self, bird_id: int, color: tuple[int, int, int]):
        self.bird_id = bird_id
        self.color = color
        self.reset()

    def reset(self):
        self.x = PLAYER_START_X
        self.y = float(PLAYER_START_Y)
        self.vel_y = float(PLAYER_FLAP_ACC)  # start with upward velocity
        self.alive = True
        self.score = 0
        self.steps_alive = 0

    def flap(self):
        if self.alive and self.y > -2 * PLAYER_HEIGHT:
            self.vel_y = float(PLAYER_FLAP_ACC)

    def update(self, ground_y: int):
        if not self.alive:
            return
        # Gravity
        if self.vel_y < PLAYER_MAX_VEL_Y:
            self.vel_y += PLAYER_ACC_Y
        # Position update (cap at ground)
        self.y += min(self.vel_y, ground_y - self.y - PLAYER_HEIGHT)
        self.y = max(self.y, 0)
        self.steps_alive += 1

    @property
    def rect(self) -> tuple[float, float, int, int]:
        return (self.x, self.y, PLAYER_WIDTH, PLAYER_HEIGHT)


class FlappyBirdEngine:
    """Multi-bird Flappy Bird engine with shared pipes."""

    def __init__(self, rng_seed: int | None = None):
        self.birds: list[Bird] = []
        self.pipes: list[dict] = []
        self.round_num = 0
        self.frame = 0
        self.rng = np.random.default_rng(rng_seed)

    def add_bird(self, color: tuple[int, int, int]) -> Bird:
        bird = Bird(bird_id=len(self.birds), color=color)
        self.birds.append(bird)
        return bird

    def remove_bird(self, bird_id: int):
        self.birds = [b for b in self.birds if b.bird_id != bird_id]
        # Reindex
        for i, b in enumerate(self.birds):
            b.bird_id = i

    def reset(self):
        """Reset all birds and pipes for a new round."""
        self.round_num += 1
        self.frame = 0
        for bird in self.birds:
            bird.reset()
        self._init_pipes()

    def _init_pipes(self):
        """Generate initial set of pipes."""
        self.pipes = []
        for i in range(3):
            x = SCREEN_WIDTH + i * (SCREEN_WIDTH // 2)
            self.pipes.append(self._make_pipe(x))

    def _make_pipe(self, x: float) -> dict:
        gap_y = int(self.rng.choice(GAP_YS)) + GAP_OFFSET
        return {
            "x": float(x),
            "gap_y": gap_y,
            "upper_y": gap_y - PIPE_HEIGHT,  # top of upper pipe
            "lower_y": gap_y + PIPE_GAP,  # top of lower pipe
            "scored": False,
        }

    def step(self, actions: dict[int, int]) -> tuple[dict, dict, dict, dict]:
        """Advance one frame.

        Args:
            actions: {bird_id: action} where action is 0 (nothing) or 1 (flap)

        Returns:
            (observations, rewards, dones, info)
        """
        self.frame += 1

        # Apply actions
        for bird_id, action in actions.items():
            if bird_id < len(self.birds) and self.birds[bird_id].alive:
                if action == 1:
                    self.birds[bird_id].flap()

        # Update bird physics
        for bird in self.birds:
            bird.update(GROUND_Y)

        # Move pipes
        for pipe in self.pipes:
            pipe["x"] += PIPE_VEL_X

        # Remove off-screen pipes and add new ones
        if self.pipes and self.pipes[0]["x"] < -PIPE_WIDTH:
            self.pipes.pop(0)
            last_x = self.pipes[-1]["x"] if self.pipes else SCREEN_WIDTH
            self.pipes.append(self._make_pipe(last_x + SCREEN_WIDTH // 2))

        # Check collisions
        self._check_collisions()

        # Check scoring
        self._check_scoring()

        # Build return dicts
        obs_dict = {}
        reward_dict = {}
        done_dict = {}
        for bird in self.birds:
            obs_dict[bird.bird_id] = self.get_observation(bird)
            done_dict[bird.bird_id] = not bird.alive

        round_over = all(not b.alive for b in self.birds)
        best_bird = max(self.birds, key=lambda b: (b.alive, b.score, b.steps_alive))

        info = {
            "round_over": round_over,
            "round_num": self.round_num,
            "frame": self.frame,
            "best_bird_id": best_bird.bird_id,
        }

        return obs_dict, reward_dict, done_dict, info

    def _check_collisions(self):
        for bird in self.birds:
            if not bird.alive:
                continue
            # Ground/ceiling collision
            if bird.y + PLAYER_HEIGHT >= GROUND_Y or bird.y <= 0:
                bird.alive = False
                continue
            # Pipe collision (rect-based)
            bx, by = bird.x, bird.y
            for pipe in self.pipes:
                px = pipe["x"]
                # Only check pipes near the bird
                if px + PIPE_WIDTH < bx or px > bx + PLAYER_WIDTH:
                    continue
                # Upper pipe
                upper_bottom = pipe["gap_y"]
                if by < upper_bottom:
                    bird.alive = False
                    break
                # Lower pipe
                lower_top = pipe["gap_y"] + PIPE_GAP
                if by + PLAYER_HEIGHT > lower_top:
                    bird.alive = False
                    break

    def _check_scoring(self):
        for pipe in self.pipes:
            if pipe["scored"]:
                continue
            pipe_mid = pipe["x"] + PIPE_WIDTH / 2
            for bird in self.birds:
                if not bird.alive:
                    continue
                bird_mid = bird.x + PLAYER_WIDTH / 2
                if bird_mid > pipe_mid:
                    bird.score += 1
            if any(b.alive and b.x + PLAYER_WIDTH / 2 > pipe_mid for b in self.birds):
                pipe["scored"] = True

    def get_observation(self, bird: Bird) -> np.ndarray:
        """Get simple observation for a bird: (y, vel, dist_pipe, gap_center).

        All values normalized to roughly [-1, 1] range.
        """
        # Find next pipe (first pipe whose right edge is ahead of bird)
        next_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + PIPE_WIDTH > bird.x:
                next_pipe = pipe
                break
        if next_pipe is None:
            next_pipe = self.pipes[-1] if self.pipes else {"x": SCREEN_WIDTH, "gap_y": GROUND_Y // 2, "lower_y": GROUND_Y // 2 + PIPE_GAP}

        player_y = bird.y / SCREEN_HEIGHT
        velocity = bird.vel_y / PLAYER_MAX_VEL_Y
        dist_next = (next_pipe["x"] - bird.x) / SCREEN_WIDTH
        gap_center = (next_pipe["gap_y"] + PIPE_GAP / 2) / SCREEN_HEIGHT

        return np.array([player_y, velocity, dist_next, gap_center], dtype=np.float32)

    def get_alive_count(self) -> int:
        return sum(1 for b in self.birds if b.alive)
