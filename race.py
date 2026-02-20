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
