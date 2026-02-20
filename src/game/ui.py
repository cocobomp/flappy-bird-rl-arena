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
