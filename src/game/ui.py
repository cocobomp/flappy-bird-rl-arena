"""Interactive UI components for the multi-bird race."""
import pygame

PANEL_WIDTH = 280
EDUCATION_WIDTH = 300
SCREEN_WIDTH = 288

BIRD_COLORS = [
    (255, 80, 80), (80, 130, 255), (80, 220, 80), (255, 200, 50),
    (200, 80, 255), (255, 140, 50), (50, 220, 220), (255, 120, 180),
]

ALGO_OPTIONS = ["q_learning", "dqn", "double_dqn"]
REWARD_OPTIONS = ["basic", "distance", "centered", "smart"]
STRATEGY_OPTIONS = ["random", "gravity", "heuristic", "guided"]

ALGO_DISPLAY = {"q_learning": "QL", "dqn": "DQN", "double_dqn": "DDQN"}
REWARD_DISPLAY = {
    "basic": "Basic", "distance": "Dist", "centered": "Center", "smart": "Smart",
}
STRATEGY_DISPLAY = {
    "random": "Random", "gravity": "Gravity",
    "heuristic": "Heurist.", "guided": "Guided",
}

STRATEGY_EXPLAIN = {
    "random": "50/50 aleatoire — monte toujours (demo)",
    "gravity": "12% de flap — compense gravite faible",
    "heuristic": "PD-controller: flap selon position+velocite",
    "guided": "Gravity loin + Heuristique pres (recommande!)",
}


class Slider:
    """Horizontal slider widget for Pygame."""

    def __init__(self, x, y, width, min_val, max_val, initial, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = 12
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.dragging = False

    def handle_event(self, event, dx=0, dy=0):
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
        ratio = max(0.0, min(1.0, (mx - self.x) / self.width))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)

    def draw(self, surface, font, dx=0, dy=0):
        if abs(self.value) < 0.01:
            val_str = f"{self.value:.4f}"
        elif abs(self.value) < 1:
            val_str = f"{self.value:.3f}"
        else:
            val_str = f"{self.value:.1f}"
        sx, sy = self.x + dx, self.y + dy
        txt = font.render(f"{self.label}: {val_str}", True, (200, 200, 200))
        surface.blit(txt, (sx, sy - 13))
        bar = pygame.Rect(sx, sy, self.width, self.height)
        pygame.draw.rect(surface, (50, 50, 65), bar, border_radius=3)
        ratio = (self.value - self.min_val) / max(1e-9, self.max_val - self.min_val)
        fill = pygame.Rect(sx, sy, max(1, int(self.width * ratio)), self.height)
        pygame.draw.rect(surface, (70, 110, 190), fill, border_radius=3)
        hx = sx + int(self.width * ratio)
        hy = sy + self.height // 2
        pygame.draw.circle(surface, (200, 200, 220), (hx, hy), 6)
        pygame.draw.circle(surface, (100, 100, 130), (hx, hy), 6, 1)


class AddBirdDialog:
    """Modal dialog: algo, reward, strategy, and 8 parameter sliders."""

    DIALOG_W = 420
    DIALOG_H = 430

    def __init__(self, window_width=868, window_height=512):
        self.active = False
        self.selected_algo = 1      # dqn
        self.selected_reward = 3    # smart
        self.selected_strategy = 3  # guided
        self.font = None
        self.font_title = None
        self.font_small = None
        self.window_width = window_width
        self.window_height = window_height
        self.dx = (window_width - self.DIALOG_W) // 2
        self.dy = (window_height - self.DIALOG_H) // 2

        sx = 15
        sw = self.DIALOG_W - 30
        # Learning params
        self.slider_epsilon = Slider(sx, 170, sw, 0.1, 1.0, 1.0, "Epsilon depart")
        self.slider_lr = Slider(sx, 196, sw, 0.0001, 0.01, 0.0003, "Learning rate")
        self.slider_decay = Slider(sx, 222, sw, 0.990, 0.99999, 0.99995, "Epsilon decay")
        # Reward params
        self.slider_death = Slider(sx, 258, sw, 1.0, 50.0, 20.0, "Penalite mort")
        self.slider_pipe = Slider(sx, 284, sw, 0.0, 50.0, 10.0, "Bonus porte")
        self.slider_alive = Slider(sx, 310, sw, 0.0, 2.0, 0.05, "Reward survie")
        # Strategy params
        self.slider_threshold = Slider(sx, 346, sw, 0.01, 0.12, 0.04, "Seuil flap")
        self.slider_noise = Slider(sx, 372, sw, 0.0, 0.30, 0.10, "Bruit strategie")

        self.sliders = [
            self.slider_epsilon, self.slider_lr, self.slider_decay,
            self.slider_death, self.slider_pipe, self.slider_alive,
            self.slider_threshold, self.slider_noise,
        ]

    def _init_fonts(self):
        if self.font is None:
            self.font = pygame.font.SysFont("monospace", 13, bold=True)
            self.font_title = pygame.font.SysFont("monospace", 16, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 10)

    def show(self):
        self.active = True
        self.selected_algo = 1
        self.selected_reward = 3
        self.selected_strategy = 3

    def hide(self):
        self.active = False

    def get_config(self) -> dict:
        return {
            "algo": ALGO_OPTIONS[self.selected_algo],
            "reward": REWARD_OPTIONS[self.selected_reward],
            "strategy": STRATEGY_OPTIONS[self.selected_strategy],
            "epsilon_start": round(self.slider_epsilon.value, 3),
            "lr": round(self.slider_lr.value, 5),
            "epsilon_decay": round(self.slider_decay.value, 4),
            "death_penalty": round(self.slider_death.value, 1),
            "pipe_bonus": round(self.slider_pipe.value, 1),
            "alive_reward": round(self.slider_alive.value, 2),
            "flap_threshold": round(self.slider_threshold.value, 3),
            "strategy_noise": round(self.slider_noise.value, 3),
        }

    def handle_event(self, event: pygame.event.Event) -> str | None:
        if not self.active:
            return None
        if event.type not in (
            pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP,
        ):
            return None

        dx, dy = self.dx, self.dy
        dw, dh = self.DIALOG_W, self.DIALOG_H

        for slider in self.sliders:
            slider.handle_event(event, dx, dy)

        if event.type != pygame.MOUSEBUTTONDOWN:
            return None

        mx, my = event.pos
        if not (dx <= mx <= dx + dw and dy <= my <= dy + dh):
            return "cancel"

        rx, ry = mx - dx, my - dy

        # Algo buttons (drawn at x=80, y=y_off+5, h=22)
        for i in range(len(ALGO_OPTIONS)):
            bx = 80 + i * 132
            if bx <= rx <= bx + 124 and 30 <= ry <= 52:
                self.selected_algo = i

        # Reward buttons
        for i in range(len(REWARD_OPTIONS)):
            bx = 80 + i * 100
            if bx <= rx <= bx + 92 and 60 <= ry <= 82:
                self.selected_reward = i

        # Strategy buttons
        for i in range(len(STRATEGY_OPTIONS)):
            bx = 80 + i * 100
            if bx <= rx <= bx + 92 and 90 <= ry <= 112:
                self.selected_strategy = i

        # Confirm (y=396..426)
        if 20 <= rx <= dw - 20 and 396 <= ry <= 426:
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
        dialog_rect = pygame.Rect(dx, dy, dw, dh)
        pygame.draw.rect(screen, (40, 40, 55), dialog_rect, border_radius=8)
        pygame.draw.rect(screen, (100, 100, 130), dialog_rect, 2, border_radius=8)

        # Title
        t = self.font_title.render("Ajouter un Oiseau", True, (0, 255, 120))
        screen.blit(t, (dx + 15, dy + 6))

        # --- Algo ---
        self._draw_button_row(screen, dx, dy, 25, "Algo:", ALGO_OPTIONS,
                              ALGO_DISPLAY, self.selected_algo, 132, 124)
        # --- Reward ---
        self._draw_button_row(screen, dx, dy, 55, "Reward:", REWARD_OPTIONS,
                              REWARD_DISPLAY, self.selected_reward, 100, 92)
        # --- Strategy ---
        self._draw_button_row(screen, dx, dy, 85, "Strat:", STRATEGY_OPTIONS,
                              STRATEGY_DISPLAY, self.selected_strategy, 100, 92)

        # Strategy explanation (single line)
        strat_key = STRATEGY_OPTIONS[self.selected_strategy]
        expl = STRATEGY_EXPLAIN.get(strat_key, "")
        et = self.font_small.render(expl, True, (140, 160, 200))
        screen.blit(et, (dx + 18, dy + 116))

        # Separator
        pygame.draw.line(screen, (80, 80, 100),
                         (dx + 15, dy + 132), (dx + dw - 15, dy + 132))

        # --- Params label ---
        lbl = self.font.render("Apprentissage:", True, (180, 180, 200))
        screen.blit(lbl, (dx + 15, dy + 136))

        # Draw first 3 sliders (learning)
        for s in self.sliders[:3]:
            s.draw(screen, self.font_small, dx, dy)

        # Reward params label
        lbl2 = self.font.render("Recompenses:", True, (180, 180, 200))
        screen.blit(lbl2, (dx + 15, dy + 240))

        # Draw reward sliders
        for s in self.sliders[3:6]:
            s.draw(screen, self.font_small, dx, dy)

        # Strategy params label
        lbl3 = self.font.render("Strategie:", True, (180, 180, 200))
        screen.blit(lbl3, (dx + 15, dy + 328))

        # Draw strategy sliders
        for s in self.sliders[6:]:
            s.draw(screen, self.font_small, dx, dy)

        # Confirm button
        cr = pygame.Rect(dx + 20, dy + 396, dw - 40, 30)
        pygame.draw.rect(screen, (40, 160, 80), cr, border_radius=6)
        pygame.draw.rect(screen, (80, 200, 120), cr, 1, border_radius=6)
        ct = self.font_title.render("AJOUTER", True, (255, 255, 255))
        screen.blit(ct, (cr.x + (cr.width - ct.get_width()) // 2, cr.y + 6))

    def _draw_button_row(self, screen, dx, dy, y_off, label, options,
                         display, selected, spacing, btn_w):
        """Draw a label + row of selection buttons."""
        lt = self.font.render(label, True, (200, 200, 200))
        screen.blit(lt, (dx + 15, dy + y_off))

        by = dy + y_off + 5
        for i, opt in enumerate(options):
            bx = dx + 80 + i * spacing
            rect = pygame.Rect(bx, by, btn_w, 22)
            color = (80, 120, 200) if i == selected else (55, 55, 75)
            pygame.draw.rect(screen, color, rect, border_radius=3)
            txt = self.font_small.render(display[opt], True, (220, 220, 220))
            screen.blit(txt, (bx + (btn_w - txt.get_width()) // 2, by + 5))
