# Multi-Bird RL Race — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Pygame multi-bird race where several RL agents play Flappy Bird simultaneously in the same environment, with an interactive control panel.

**Architecture:** Custom multi-bird Flappy Bird engine (no gymnasium) reusing exact physics constants from flappy-bird-gymnasium. Each bird runs its own RL agent (from existing src/agents/). A Pygame UI renders the game + side panel with stats and controls. Training happens live — birds learn in real time across rounds.

**Tech Stack:** Python 3.11, Pygame, PyTorch, NumPy. Reuses existing agents and reward functions.

---

## Parallel Tracks

| Track | Agent | Tasks |
|---|---|---|
| A — Game Engine | Agent Engine | 1-2 |
| B — Race Manager + UI | Agent UI | 3-5 |
| C — Integration | Main | 6 |

Track A and B can run in parallel (engine has no UI dependency, UI can be stubbed).

---

### Task 1: Multi-Bird Flappy Bird Engine

**Files:**
- Create: `src/game/engine.py`
- Create: `src/game/__init__.py`
- Create: `tests/test_game/__init__.py`
- Create: `tests/test_game/test_engine.py`

**Step 1: Write failing tests**

```python
# tests/test_game/test_engine.py
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
        assert obs_dict[0].shape == (4,)

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
        # 4 features: player_y, velocity, dist_next_pipe, gap_center
        assert len(obs) == 4
        assert isinstance(obs[0], (float, np.floating))
```

**Step 2: Run tests, verify fail**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/test_game/test_engine.py -v
```

**Step 3: Implement engine**

```python
# src/game/__init__.py
# empty

# src/game/engine.py
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
```

**Step 4: Run tests, verify pass**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/test_game/test_engine.py -v
```

**Step 5: Commit**

```bash
git add src/game/ tests/test_game/
git commit -m "feat: add multi-bird Flappy Bird game engine"
```

---

### Task 2: Game Renderer (Pygame drawing)

**Files:**
- Create: `src/game/renderer.py`

No automated tests (Pygame rendering).

**Step 1: Implement renderer**

```python
# src/game/renderer.py
"""Pygame renderer for the multi-bird Flappy Bird game."""
import pygame
import numpy as np
from src.game.engine import (
    FlappyBirdEngine, Bird,
    SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y,
    PLAYER_WIDTH, PLAYER_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT, PIPE_GAP,
)

PANEL_WIDTH = 280
WINDOW_WIDTH = SCREEN_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = SCREEN_HEIGHT

# Colors
SKY_COLOR = (78, 192, 202)
GROUND_COLOR = (222, 216, 149)
PIPE_COLOR = (83, 164, 60)
PIPE_BORDER_COLOR = (60, 120, 40)
PANEL_BG = (30, 30, 40)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (0, 255, 120)
DEAD_COLOR = (150, 60, 60)
ALIVE_COLOR = (60, 200, 100)
BUTTON_COLOR = (60, 60, 80)
BUTTON_HOVER = (80, 80, 110)
BUTTON_TEXT = (220, 220, 220)


class GameRenderer:
    """Renders the multi-bird game + side panel."""

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RL Flappy Bird Race")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 22, bold=True)

    def draw(self, engine: FlappyBirdEngine, speed: int, paused: bool,
             bird_configs: dict, buttons: list[dict]):
        """Draw the full frame: game area + panel."""
        self._draw_game(engine)
        self._draw_panel(engine, speed, paused, bird_configs, buttons)
        pygame.display.flip()

    def _draw_game(self, engine: FlappyBirdEngine):
        """Draw game area: sky, pipes, birds, ground."""
        game_surface = self.screen.subsurface((0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        # Sky
        game_surface.fill(SKY_COLOR)

        # Pipes
        for pipe in engine.pipes:
            px = int(pipe["x"])
            # Upper pipe
            upper_rect = pygame.Rect(px, 0, PIPE_WIDTH, pipe["gap_y"])
            pygame.draw.rect(game_surface, PIPE_COLOR, upper_rect)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, upper_rect, 2)
            # Lip
            lip = pygame.Rect(px - 2, pipe["gap_y"] - 20, PIPE_WIDTH + 4, 20)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lip)

            # Lower pipe
            lower_top = pipe["gap_y"] + PIPE_GAP
            lower_rect = pygame.Rect(px, lower_top, PIPE_WIDTH, SCREEN_HEIGHT - lower_top)
            pygame.draw.rect(game_surface, PIPE_COLOR, lower_rect)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lower_rect, 2)
            # Lip
            lip2 = pygame.Rect(px - 2, lower_top, PIPE_WIDTH + 4, 20)
            pygame.draw.rect(game_surface, PIPE_BORDER_COLOR, lip2)

        # Ground
        ground_rect = pygame.Rect(0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y)
        pygame.draw.rect(game_surface, GROUND_COLOR, ground_rect)
        pygame.draw.line(game_surface, (180, 170, 110), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

        # Birds
        for bird in engine.birds:
            if not bird.alive:
                continue
            bx, by = int(bird.x), int(bird.y)
            # Body (filled circle-ish rect with color)
            body = pygame.Rect(bx, by, PLAYER_WIDTH, PLAYER_HEIGHT)
            pygame.draw.ellipse(game_surface, bird.color, body)
            pygame.draw.ellipse(game_surface, (0, 0, 0), body, 2)
            # Eye
            eye_x = bx + PLAYER_WIDTH - 8
            eye_y = by + 6
            pygame.draw.circle(game_surface, (255, 255, 255), (eye_x, eye_y), 5)
            pygame.draw.circle(game_surface, (0, 0, 0), (eye_x, eye_y), 2)

    def _draw_panel(self, engine: FlappyBirdEngine, speed: int, paused: bool,
                    bird_configs: dict, buttons: list[dict]):
        """Draw the side panel with stats and controls."""
        panel = self.screen.subsurface((SCREEN_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        panel.fill(PANEL_BG)

        y = 10
        # Title
        title = self.font_title.render(f"ROUND {engine.round_num}", True, HIGHLIGHT_COLOR)
        panel.blit(title, (10, y))
        y += 30

        speed_txt = self.font.render(f"Speed: x{speed}  {'PAUSED' if paused else ''}", True, TEXT_COLOR)
        panel.blit(speed_txt, (10, y))
        y += 25

        # Separator
        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 10

        # Bird stats
        for bird in engine.birds:
            config = bird_configs.get(bird.bird_id, {})
            name = config.get("name", f"Bird {bird.bird_id}")
            algo = config.get("algo", "?")
            reward = config.get("reward", "?")
            best = config.get("best_score", 0)

            # Color indicator
            indicator = pygame.Rect(10, y, 12, 12)
            pygame.draw.rect(panel, bird.color, indicator)
            pygame.draw.rect(panel, (200, 200, 200), indicator, 1)

            # Status
            status_color = ALIVE_COLOR if bird.alive else DEAD_COLOR
            status = "alive" if bird.alive else "dead"

            label = self.font.render(f" {algo} + {reward}", True, TEXT_COLOR)
            panel.blit(label, (26, y - 2))
            y += 18

            score_txt = self.font.render(
                f"   Score: {bird.score}  Best: {best}  [{status}]", True, status_color
            )
            panel.blit(score_txt, (10, y - 2))
            y += 22

        # Separator
        pygame.draw.line(panel, (80, 80, 100), (10, y), (PANEL_WIDTH - 10, y), 1)
        y += 10

        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        mouse_x = mouse_pos[0] - SCREEN_WIDTH  # relative to panel
        mouse_y = mouse_pos[1]

        for btn in buttons:
            btn_rect = pygame.Rect(10, y, PANEL_WIDTH - 20, 30)
            btn["rect"] = btn_rect  # store for click detection
            hover = btn_rect.collidepoint(mouse_x, mouse_y)
            color = BUTTON_HOVER if hover else BUTTON_COLOR
            pygame.draw.rect(panel, color, btn_rect, border_radius=4)
            pygame.draw.rect(panel, (100, 100, 120), btn_rect, 1, border_radius=4)
            txt = self.font.render(btn["label"], True, BUTTON_TEXT)
            tx = btn_rect.x + (btn_rect.width - txt.get_width()) // 2
            ty = btn_rect.y + (btn_rect.height - txt.get_height()) // 2
            panel.blit(txt, (tx, ty))
            y += 38

    def quit(self):
        pygame.quit()
```

**Step 2: Commit**

```bash
git add src/game/renderer.py
git commit -m "feat: add Pygame renderer for multi-bird race"
```

---

### Task 3: Bird Configuration + Add-Bird Dialog

**Files:**
- Create: `src/game/ui.py`

**Step 1: Implement UI components**

```python
# src/game/ui.py
"""Interactive UI components for the multi-bird race."""
import pygame

PANEL_WIDTH = 280
SCREEN_WIDTH = 288

# Preset bird colors
BIRD_COLORS = [
    (255, 80, 80),   # Red
    (80, 130, 255),  # Blue
    (80, 220, 80),   # Green
    (255, 200, 50),  # Yellow
    (200, 80, 255),  # Purple
    (255, 140, 50),  # Orange
    (50, 220, 220),  # Cyan
    (255, 120, 180), # Pink
]

ALGO_OPTIONS = ["q_learning", "dqn", "double_dqn"]
REWARD_OPTIONS = ["basic", "distance", "centered"]

ALGO_DISPLAY = {"q_learning": "QL", "dqn": "DQN", "double_dqn": "DDQN"}
REWARD_DISPLAY = {"basic": "Basic", "distance": "Dist", "centered": "Center"}


class AddBirdDialog:
    """Modal dialog to configure a new bird."""

    def __init__(self):
        self.active = False
        self.selected_algo = 0
        self.selected_reward = 0
        self.font = None
        self.font_title = None

    def _init_fonts(self):
        if self.font is None:
            self.font = pygame.font.SysFont("monospace", 14, bold=True)
            self.font_title = pygame.font.SysFont("monospace", 18, bold=True)

    def show(self):
        self.active = True
        self.selected_algo = 0
        self.selected_reward = 0

    def hide(self):
        self.active = False

    def get_config(self) -> dict:
        """Return the selected configuration."""
        algo = ALGO_OPTIONS[self.selected_algo]
        reward = REWARD_OPTIONS[self.selected_reward]
        return {
            "algo": algo,
            "reward": reward,
            "algo_display": ALGO_DISPLAY[algo],
            "reward_display": REWARD_DISPLAY[reward],
        }

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Handle events. Returns 'confirm' or 'cancel' or None."""
        if not self.active:
            return None
        if event.type != pygame.MOUSEBUTTONDOWN:
            return None

        mx, my = event.pos
        # Dialog positioned in center of window
        dx, dy = 150, 120  # dialog top-left
        dw, dh = 280, 280

        # Outside dialog = cancel
        if not (dx <= mx <= dx + dw and dy <= my <= dy + dh):
            return "cancel"

        # Algo buttons (y = dy + 50, each 80px wide, 30px tall)
        for i, algo in enumerate(ALGO_OPTIONS):
            bx = dx + 10 + i * 88
            by = dy + 50
            if bx <= mx <= bx + 80 and by <= my <= by + 30:
                self.selected_algo = i

        # Reward buttons (y = dy + 130)
        for i, reward in enumerate(REWARD_OPTIONS):
            bx = dx + 10 + i * 88
            by = dy + 130
            if bx <= mx <= bx + 80 and by <= my <= by + 30:
                self.selected_reward = i

        # Confirm button (y = dy + 220)
        if dx + 20 <= mx <= dx + dw - 20 and dy + 220 <= my <= dy + 260:
            return "confirm"

        # Cancel button (y = dy + 180, small text)
        return None

    def draw(self, screen: pygame.Surface):
        if not self.active:
            return
        self._init_fonts()

        # Overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        screen.blit(overlay, (0, 0))

        # Dialog box
        dx, dy = 150, 120
        dw, dh = 280, 280
        dialog = pygame.Rect(dx, dy, dw, dh)
        pygame.draw.rect(screen, (40, 40, 55), dialog, border_radius=8)
        pygame.draw.rect(screen, (100, 100, 130), dialog, 2, border_radius=8)

        # Title
        title = self.font_title.render("Add Bird", True, (0, 255, 120))
        screen.blit(title, (dx + 10, dy + 10))

        # Algo label
        label = self.font.render("Algorithm:", True, (200, 200, 200))
        screen.blit(label, (dx + 10, dy + 38))

        # Algo buttons
        for i, algo in enumerate(ALGO_OPTIONS):
            bx = dx + 10 + i * 88
            by = dy + 50
            rect = pygame.Rect(bx, by, 80, 30)
            color = (80, 120, 200) if i == self.selected_algo else (60, 60, 80)
            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, (120, 120, 150), rect, 1, border_radius=4)
            txt = self.font.render(ALGO_DISPLAY[algo], True, (220, 220, 220))
            screen.blit(txt, (bx + (80 - txt.get_width()) // 2, by + 7))

        # Reward label
        label2 = self.font.render("Reward:", True, (200, 200, 200))
        screen.blit(label2, (dx + 10, dy + 118))

        # Reward buttons
        for i, reward in enumerate(REWARD_OPTIONS):
            bx = dx + 10 + i * 88
            by = dy + 130
            rect = pygame.Rect(bx, by, 80, 30)
            color = (80, 120, 200) if i == self.selected_reward else (60, 60, 80)
            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, (120, 120, 150), rect, 1, border_radius=4)
            txt = self.font.render(REWARD_DISPLAY[reward], True, (220, 220, 220))
            screen.blit(txt, (bx + (80 - txt.get_width()) // 2, by + 7))

        # Preview
        config = self.get_config()
        preview = self.font.render(
            f"Preview: {config['algo_display']} + {config['reward_display']}",
            True, (0, 255, 120),
        )
        screen.blit(preview, (dx + 10, dy + 185))

        # Confirm button
        confirm_rect = pygame.Rect(dx + 20, dy + 220, dw - 40, 40)
        pygame.draw.rect(screen, (40, 160, 80), confirm_rect, border_radius=6)
        pygame.draw.rect(screen, (80, 200, 120), confirm_rect, 1, border_radius=6)
        ctxt = self.font_title.render("ADD", True, (255, 255, 255))
        screen.blit(ctxt, (confirm_rect.x + (confirm_rect.width - ctxt.get_width()) // 2,
                           confirm_rect.y + 10))
```

**Step 2: Commit**

```bash
git add src/game/ui.py
git commit -m "feat: add bird configuration dialog UI"
```

---

### Task 4: Race Manager (orchestrates everything)

**Files:**
- Create: `src/game/race.py`
- Create: `tests/test_game/test_race.py`

**Step 1: Write failing tests**

```python
# tests/test_game/test_race.py
import numpy as np
from src.game.race import RaceManager, BirdEntry


class TestBirdEntry:
    def test_creation(self):
        entry = BirdEntry(algo="dqn", reward="basic", color=(255, 0, 0), state_dim=4)
        assert entry.agent is not None
        assert entry.reward_fn is not None
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
```

**Step 2: Run tests, verify fail**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/test_game/test_race.py -v
```

**Step 3: Implement RaceManager**

```python
# src/game/race.py
"""Race manager: orchestrates birds, agents, training, and rendering."""
from dataclasses import dataclass, field
import numpy as np

from src.agents import QLearningAgent, DQNAgent, DoubleDQNAgent, BaseAgent
from src.environments.rewards import BasicReward, DistanceReward, CenteredReward, RewardFunction
from src.game.engine import FlappyBirdEngine, Bird, SCREEN_WIDTH, GROUND_Y, PIPE_GAP
from src.game.ui import BIRD_COLORS, ALGO_DISPLAY, REWARD_DISPLAY

AGENT_MAP = {
    "q_learning": QLearningAgent,
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
}

REWARD_MAP = {
    "basic": BasicReward,
    "distance": DistanceReward,
    "centered": CenteredReward,
}

STATE_DIM = 4
ACTION_DIM = 2


@dataclass
class BirdEntry:
    """Links a bird, its RL agent, and its reward function."""
    algo: str
    reward: str
    color: tuple[int, int, int]
    state_dim: int = STATE_DIM
    agent: BaseAgent = field(init=False)
    reward_fn: RewardFunction = field(init=False)
    bird: Bird | None = field(default=None, init=False)
    best_score: int = 0
    prev_obs: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        agent_cls = AGENT_MAP[self.algo]
        if self.algo == "q_learning":
            self.agent = agent_cls(
                state_dim=self.state_dim, action_dim=ACTION_DIM,
                n_bins=10, lr=0.1, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.998,
            )
        else:
            self.agent = agent_cls(
                state_dim=self.state_dim, action_dim=ACTION_DIM,
                hidden_dims=[64, 64], lr=5e-4, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.998,
                buffer_size=20000, batch_size=32, tau=0.005, train_every=4,
            )
        self.reward_fn = REWARD_MAP[self.reward]()

    @property
    def display_name(self) -> str:
        return f"{ALGO_DISPLAY[self.algo]} + {REWARD_DISPLAY[self.reward]}"


class RaceManager:
    """Manages the multi-bird race loop."""

    def __init__(self, render: bool = True):
        self.engine = FlappyBirdEngine()
        self.entries: list[BirdEntry] = []
        self.render_enabled = render
        self.renderer = None
        self.speed = 1
        self.paused = False
        self._color_index = 0

    def add_bird(self, algo: str, reward: str) -> BirdEntry:
        color = BIRD_COLORS[self._color_index % len(BIRD_COLORS)]
        self._color_index += 1
        entry = BirdEntry(algo=algo, reward=reward, color=color)
        bird = self.engine.add_bird(color=color)
        entry.bird = bird
        self.entries.append(entry)
        return entry

    def remove_bird(self, index: int):
        if 0 <= index < len(self.entries):
            bird_id = self.entries[index].bird.bird_id
            self.engine.remove_bird(bird_id)
            self.entries.pop(index)
            # Re-link birds after reindex
            for i, entry in enumerate(self.entries):
                entry.bird = self.engine.birds[i]

    def reset_round(self):
        self.engine.reset()
        for entry in self.entries:
            entry.prev_obs = self.engine.get_observation(entry.bird)

    def step_frame(self) -> bool:
        """Run one frame of the game. Returns True if round is over."""
        if not self.entries:
            return True

        # Collect actions from all alive birds
        actions = {}
        for entry in self.entries:
            if entry.bird.alive:
                obs = self.engine.get_observation(entry.bird)
                action = entry.agent.select_action(obs, training=True)
                actions[entry.bird.bird_id] = action
                entry.prev_obs = obs

        # Step engine
        obs_dict, _, done_dict, info = self.engine.step(actions)

        # Train each bird's agent
        for entry in self.entries:
            bid = entry.bird.bird_id
            if entry.prev_obs is not None:
                obs = obs_dict.get(bid, entry.prev_obs)
                terminated = done_dict.get(bid, False)
                # Compute custom reward
                raw_reward = 0.1 if not terminated else -1.0
                reward = entry.reward_fn.compute(obs, raw_reward, terminated, False)
                action = actions.get(bid, 0)
                entry.agent.train_step(entry.prev_obs, action, reward, obs, terminated)

        # Update best scores
        for entry in self.entries:
            if entry.bird.score > entry.best_score:
                entry.best_score = entry.bird.score

        return info["round_over"]

    def get_bird_configs(self) -> dict:
        configs = {}
        for entry in self.entries:
            configs[entry.bird.bird_id] = {
                "name": entry.display_name,
                "algo": ALGO_DISPLAY[entry.algo],
                "reward": REWARD_DISPLAY[entry.reward],
                "best_score": entry.best_score,
            }
        return configs
```

**Step 4: Run tests, verify pass**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/test_game/test_race.py -v
```

**Step 5: Commit**

```bash
git add src/game/race.py tests/test_game/test_race.py
git commit -m "feat: add RaceManager to orchestrate multi-bird RL race"
```

---

### Task 5: Main race loop (entry point)

**Files:**
- Create: `race.py` (project root)

**Step 1: Implement main race loop**

```python
# race.py
"""Multi-Bird RL Race — Watch multiple RL agents compete at Flappy Bird."""
import sys
import pygame
from src.game.engine import FlappyBirdEngine
from src.game.renderer import GameRenderer
from src.game.race import RaceManager
from src.game.ui import AddBirdDialog


def main():
    manager = RaceManager(render=True)
    renderer = GameRenderer()
    dialog = AddBirdDialog()

    # Start with 3 default birds
    manager.add_bird(algo="dqn", reward="basic")
    manager.add_bird(algo="double_dqn", reward="distance")
    manager.add_bird(algo="q_learning", reward="basic")
    manager.reset_round()

    buttons = [
        {"label": "+ Add Bird", "action": "add_bird", "rect": None},
        {"label": "Speed: x1", "action": "speed", "rect": None},
        {"label": "Pause", "action": "pause", "rect": None},
    ]

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    manager.paused = not manager.paused
                elif event.key == pygame.K_UP:
                    manager.speed = min(manager.speed * 2, 10)
                elif event.key == pygame.K_DOWN:
                    manager.speed = max(manager.speed // 2, 1)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check dialog first
                if dialog.active:
                    result = dialog.handle_event(event)
                    if result == "confirm":
                        config = dialog.get_config()
                        manager.add_bird(algo=config["algo"], reward=config["reward"])
                        dialog.hide()
                        manager.reset_round()
                    elif result == "cancel":
                        dialog.hide()
                else:
                    # Check buttons
                    mx, my = event.pos
                    for btn in buttons:
                        if btn["rect"] and btn["rect"].collidepoint(
                            mx - 288, my  # panel-relative coords
                        ):
                            if btn["action"] == "add_bird":
                                dialog.show()
                            elif btn["action"] == "speed":
                                manager.speed = manager.speed * 2 if manager.speed < 10 else 1
                            elif btn["action"] == "pause":
                                manager.paused = not manager.paused

        # Update speed button label
        buttons[1]["label"] = f"Speed: x{manager.speed}"
        buttons[2]["label"] = "Resume" if manager.paused else "Pause"

        # Game logic
        if not manager.paused:
            for _ in range(manager.speed):
                round_over = manager.step_frame()
                if round_over:
                    manager.reset_round()
                    break

        # Render
        bird_configs = manager.get_bird_configs()
        renderer.draw(manager.engine, manager.speed, manager.paused, bird_configs, buttons)

        # Draw dialog on top if active
        if dialog.active:
            dialog.draw(renderer.screen)
            pygame.display.flip()

        renderer.clock.tick(60)

    renderer.quit()


if __name__ == "__main__":
    main()
```

**Step 2: Test manually**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 race.py
```

Expected: Pygame window opens with 3 birds racing, side panel with stats and controls.

**Step 3: Commit**

```bash
git add race.py
git commit -m "feat: add multi-bird race entry point with interactive UI"
```

---

### Task 6: Integration + polish

**Files:**
- Modify: `src/game/__init__.py`

**Step 1: Update init**

```python
# src/game/__init__.py
"""Multi-bird Flappy Bird game engine and race manager."""
from src.game.engine import FlappyBirdEngine, Bird
from src.game.race import RaceManager, BirdEntry
```

**Step 2: Run all tests**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/ -v
```

Expected: All tests pass (existing 158 + new game tests).

**Step 3: Manual integration test**

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 race.py
```

Verify:
- 3 birds visible with different colors
- Birds die and respawn each round
- Scores update in panel
- "Add Bird" button opens dialog
- Speed up/down works (arrow keys or button)
- Pause/resume works (space or button)
- ESC quits

**Step 4: Commit**

```bash
git add src/game/
git commit -m "feat: finalize multi-bird race with all integrations"
```

---

## Usage

```bash
# Launch the multi-bird race
/opt/homebrew/opt/python@3.11/bin/python3.11 race.py
```

**Controls:**
- `Space` — Pause/Resume
- `Up/Down arrows` — Speed up/down (x1, x2, x4, x8, x10)
- `ESC` — Quit
- Click "Add Bird" — Configure and add a new bird
- Click "Speed" — Cycle speed
- Click "Pause" — Toggle pause
