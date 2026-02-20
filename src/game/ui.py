"""Interactive UI components for the multi-bird race."""
import pygame

PANEL_WIDTH = 280
EDUCATION_WIDTH = 300
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
REWARD_OPTIONS = ["basic", "distance", "centered", "smart"]

ALGO_DISPLAY = {"q_learning": "QL", "dqn": "DQN", "double_dqn": "DDQN"}
REWARD_DISPLAY = {
    "basic": "Basic", "distance": "Dist", "centered": "Center", "smart": "Smart",
}

ALGO_EXPLAIN = {
    "q_learning": [
        "Table de Q-valeurs pour chaque",
        "etat discretise. Simple et",
        "interpretable mais limite.",
    ],
    "dqn": [
        "Reseau de neurones qui approxime",
        "les Q-valeurs. Experience replay",
        "et target network stabilisent",
        "l'apprentissage.",
    ],
    "double_dqn": [
        "Separe selection et evaluation",
        "pour reduire la surestimation",
        "des Q-valeurs du DQN.",
    ],
}

REWARD_EXPLAIN = {
    "basic": ["+1 en vie, -1000 a la mort.", "Signal sparse."],
    "distance": ["Bonus de proximite au", "prochain tuyau."],
    "centered": ["Bonus pour rester centre", "dans le gap du tuyau."],
    "smart": [
        "Combine centrage + direction",
        "+ progression. Recommande !",
    ],
}


class Slider:
    """Simple horizontal slider widget for Pygame."""

    def __init__(self, x, y, width, min_val, max_val, initial, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = 14
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.dragging = False

    def handle_event(self, event, dx=0, dy=0):
        """Handle mouse event. dx, dy are parent offset."""
        mx = event.pos[0] - dx
        my = event.pos[1] - dy
        bar = pygame.Rect(self.x, self.y, self.width, self.height)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if bar.inflate(0, 16).collidepoint(mx, my):
                self.dragging = True
                self._update(mx)
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update(mx)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

    def _update(self, mx):
        ratio = (mx - self.x) / self.width
        ratio = max(0.0, min(1.0, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)

    def draw(self, surface, font, dx=0, dy=0):
        # Format value
        if abs(self.value) < 0.01:
            val_str = f"{self.value:.4f}"
        elif abs(self.value) < 1:
            val_str = f"{self.value:.3f}"
        else:
            val_str = f"{self.value:.1f}"

        sx = self.x + dx
        sy = self.y + dy

        # Label
        txt = font.render(f"{self.label}: {val_str}", True, (200, 200, 200))
        surface.blit(txt, (sx, sy - 14))

        # Bar background
        bar = pygame.Rect(sx, sy, self.width, self.height)
        pygame.draw.rect(surface, (50, 50, 65), bar, border_radius=3)

        # Fill
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill = pygame.Rect(sx, sy, max(1, int(self.width * ratio)), self.height)
        pygame.draw.rect(surface, (70, 110, 190), fill, border_radius=3)

        # Handle
        hx = sx + int(self.width * ratio)
        hy = sy + self.height // 2
        pygame.draw.circle(surface, (200, 200, 220), (hx, hy), 7)
        pygame.draw.circle(surface, (100, 100, 130), (hx, hy), 7, 1)


class AddBirdDialog:
    """Modal dialog to configure a new bird with algo, reward, and hyperparams."""

    DIALOG_W = 420
    DIALOG_H = 470

    def __init__(self, window_width=868, window_height=512):
        self.active = False
        self.selected_algo = 1  # default: dqn
        self.selected_reward = 3  # default: smart
        self.font = None
        self.font_title = None
        self.font_small = None
        self.window_width = window_width
        self.window_height = window_height

        # Dialog position (centered)
        self.dx = (window_width - self.DIALOG_W) // 2
        self.dy = (window_height - self.DIALOG_H) // 2

        # Sliders (positions relative to dialog)
        sx = 15
        sw = self.DIALOG_W - 30
        self.slider_epsilon = Slider(sx, 260, sw, 0.1, 1.0, 0.8, "Epsilon depart")
        self.slider_lr = Slider(sx, 300, sw, 0.0001, 0.01, 0.0005, "Learning rate")
        self.slider_decay = Slider(sx, 340, sw, 0.990, 0.999, 0.998, "Epsilon decay")
        self.slider_death = Slider(sx, 380, sw, 1.0, 20.0, 5.0, "Penalite mort")
        self.sliders = [
            self.slider_epsilon, self.slider_lr,
            self.slider_decay, self.slider_death,
        ]

    def _init_fonts(self):
        if self.font is None:
            self.font = pygame.font.SysFont("monospace", 14, bold=True)
            self.font_title = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 11)

    def show(self):
        self.active = True
        self.selected_algo = 1
        self.selected_reward = 3

    def hide(self):
        self.active = False

    def get_config(self) -> dict:
        algo = ALGO_OPTIONS[self.selected_algo]
        reward = REWARD_OPTIONS[self.selected_reward]
        return {
            "algo": algo,
            "reward": reward,
            "epsilon_start": round(self.slider_epsilon.value, 3),
            "lr": round(self.slider_lr.value, 5),
            "epsilon_decay": round(self.slider_decay.value, 4),
            "death_penalty": round(self.slider_death.value, 1),
        }

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Handle mouse events. Returns 'confirm', 'cancel', or None."""
        if not self.active:
            return None
        if event.type not in (
            pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP,
        ):
            return None

        dx, dy = self.dx, self.dy
        dw, dh = self.DIALOG_W, self.DIALOG_H

        # Forward all mouse events to sliders
        for slider in self.sliders:
            slider.handle_event(event, dx, dy)

        # Only process clicks for buttons
        if event.type != pygame.MOUSEBUTTONDOWN:
            return None

        mx, my = event.pos

        # Outside dialog = cancel
        if not (dx <= mx <= dx + dw and dy <= my <= dy + dh):
            return "cancel"

        # Relative to dialog
        rx = mx - dx
        ry = my - dy

        # Algo buttons (y=48, 3 buttons)
        for i in range(len(ALGO_OPTIONS)):
            bx = 15 + i * 132
            if bx <= rx <= bx + 124 and 48 <= ry <= 78:
                self.selected_algo = i

        # Reward buttons (y=150, 4 buttons)
        for i in range(len(REWARD_OPTIONS)):
            bx = 15 + i * 100
            if bx <= rx <= bx + 92 and 150 <= ry <= 180:
                self.selected_reward = i

        # Confirm button (y=415)
        if 20 <= rx <= dw - 20 and 415 <= ry <= 455:
            return "confirm"

        return None

    def draw(self, screen: pygame.Surface):
        if not self.active:
            return
        self._init_fonts()

        dx, dy = self.dx, self.dy
        dw, dh = self.DIALOG_W, self.DIALOG_H

        # Overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        screen.blit(overlay, (0, 0))

        # Dialog box
        dialog = pygame.Rect(dx, dy, dw, dh)
        pygame.draw.rect(screen, (40, 40, 55), dialog, border_radius=8)
        pygame.draw.rect(screen, (100, 100, 130), dialog, 2, border_radius=8)

        # Title
        title = self.font_title.render("Ajouter un Oiseau", True, (0, 255, 120))
        screen.blit(title, (dx + 15, dy + 10))

        # --- Algorithm section ---
        lbl = self.font.render("Algorithme:", True, (200, 200, 200))
        screen.blit(lbl, (dx + 15, dy + 35))

        for i, algo in enumerate(ALGO_OPTIONS):
            bx = dx + 15 + i * 132
            by = dy + 48
            rect = pygame.Rect(bx, by, 124, 30)
            color = (80, 120, 200) if i == self.selected_algo else (60, 60, 80)
            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, (120, 120, 150), rect, 1, border_radius=4)
            txt = self.font.render(ALGO_DISPLAY[algo], True, (220, 220, 220))
            screen.blit(txt, (bx + (124 - txt.get_width()) // 2, by + 7))

        # Algo explanation
        algo_key = ALGO_OPTIONS[self.selected_algo]
        ey = dy + 82
        for line in ALGO_EXPLAIN.get(algo_key, []):
            t = self.font_small.render(line, True, (150, 170, 200))
            screen.blit(t, (dx + 20, ey))
            ey += 13

        # Separator
        pygame.draw.line(screen, (80, 80, 100), (dx + 15, dy + 135), (dx + dw - 15, dy + 135))

        # --- Reward section ---
        lbl2 = self.font.render("Recompense:", True, (200, 200, 200))
        screen.blit(lbl2, (dx + 15, dy + 138))

        for i, reward in enumerate(REWARD_OPTIONS):
            bx = dx + 15 + i * 100
            by = dy + 150
            rect = pygame.Rect(bx, by, 92, 30)
            color = (80, 120, 200) if i == self.selected_reward else (60, 60, 80)
            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, (120, 120, 150), rect, 1, border_radius=4)
            txt = self.font.render(REWARD_DISPLAY[reward], True, (220, 220, 220))
            screen.blit(txt, (bx + (92 - txt.get_width()) // 2, by + 7))

        # Reward explanation
        rew_key = REWARD_OPTIONS[self.selected_reward]
        ey = dy + 184
        for line in REWARD_EXPLAIN.get(rew_key, []):
            t = self.font_small.render(line, True, (150, 170, 200))
            screen.blit(t, (dx + 20, ey))
            ey += 13

        # Separator
        pygame.draw.line(screen, (80, 80, 100), (dx + 15, dy + 220), (dx + dw - 15, dy + 220))

        # --- Hyperparameters ---
        lbl3 = self.font.render("Hyperparametres:", True, (200, 200, 200))
        screen.blit(lbl3, (dx + 15, dy + 225))

        for slider in self.sliders:
            slider.draw(screen, self.font_small, dx, dy)

        # --- Confirm button ---
        confirm_rect = pygame.Rect(dx + 20, dy + 415, dw - 40, 40)
        pygame.draw.rect(screen, (40, 160, 80), confirm_rect, border_radius=6)
        pygame.draw.rect(screen, (80, 200, 120), confirm_rect, 1, border_radius=6)
        ctxt = self.font_title.render("AJOUTER", True, (255, 255, 255))
        screen.blit(ctxt, (
            confirm_rect.x + (confirm_rect.width - ctxt.get_width()) // 2,
            confirm_rect.y + 10,
        ))
