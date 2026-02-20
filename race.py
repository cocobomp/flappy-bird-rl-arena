"""Multi-Bird RL Race — Watch multiple RL agents compete at Flappy Bird."""
import pygame
from src.game.renderer import GameRenderer, WINDOW_WIDTH, WINDOW_HEIGHT
from src.game.race import RaceManager
from src.game.ui import AddBirdDialog


def main():
    manager = RaceManager(render=True)
    renderer = GameRenderer()
    dialog = AddBirdDialog(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT)

    # Start with 3 default birds showcasing different strategies
    manager.add_bird(algo="dqn", reward="smart", strategy="guided")
    manager.add_bird(algo="dqn", reward="smart", strategy="random")
    manager.add_bird(algo="dqn", reward="smart", strategy="heuristic")
    manager.reset_round()

    buttons = [
        {"label": "+ Add Bird", "action": "add_bird", "rect": None},
        {"label": "- Remove Bird", "action": "remove_bird", "rect": None},
        {"label": "Speed: x1", "action": "speed", "rect": None},
        {"label": "Pause", "action": "pause", "rect": None},
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    manager.paused = not manager.paused
                elif event.key == pygame.K_UP:
                    manager.speed = min(manager.speed * 2, 64)
                elif event.key == pygame.K_DOWN:
                    manager.speed = max(manager.speed // 2, 1)

            # Forward mouse events to dialog (needed for slider dragging)
            elif event.type in (
                pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP,
            ):
                if dialog.active:
                    result = dialog.handle_event(event)
                    if result == "confirm":
                        config = dialog.get_config()
                        manager.add_bird(**config)
                        dialog.hide()
                        manager.reset_round()
                    elif result == "cancel":
                        dialog.hide()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    for btn in buttons:
                        if btn["rect"] and btn["rect"].collidepoint(
                            mx - 288, my
                        ):
                            if btn["action"] == "add_bird":
                                dialog.show()
                            elif btn["action"] == "remove_bird":
                                if manager.entries:
                                    manager.remove_bird(len(manager.entries) - 1)
                                    if manager.entries:
                                        manager.reset_round()
                            elif btn["action"] == "speed":
                                if manager.speed < 64:
                                    manager.speed *= 2
                                else:
                                    manager.speed = 1
                            elif btn["action"] == "pause":
                                manager.paused = not manager.paused

        # Update button labels
        buttons[2]["label"] = f"Speed: x{manager.speed}"
        buttons[3]["label"] = "Resume" if manager.paused else "Pause"

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
