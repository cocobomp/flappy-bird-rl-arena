"""Multi-Bird RL Race — Watch multiple RL agents compete at Flappy Bird."""
import pygame
from src.game.renderer import GameRenderer, WINDOW_WIDTH, WINDOW_HEIGHT
from src.game.race import RaceManager
from src.game.ui import AddBirdDialog


def main():
    manager = RaceManager(render=True)
    renderer = GameRenderer()
    dialog = AddBirdDialog(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT)

    # Start with one bird per algorithm to compare them all
    for algo in ("q_learning", "dqn", "double_dqn", "dueling_dqn", "reinforce", "ppo"):
        manager.add_bird(algo=algo, reward="smart", strategy="guided")
    manager.evolution_enabled = False
    manager.reset_round()

    buttons = [
        {"label": "+ Add Bird", "action": "add_bird", "rect": None},
        {"label": "Boost All (e=0.05)", "action": "boost", "rect": None},
        {"label": "Evolution: OFF", "action": "evolution", "rect": None},
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
                    renderer.speed = min(renderer.speed * 2, 64)
                elif event.key == pygame.K_DOWN:
                    renderer.speed = max(renderer.speed // 2, 1)
                elif event.key == pygame.K_b:
                    manager.boost_all()

            # Forward all mouse events to renderer + dialog
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
                else:
                    # Panel interactive controls (speed slider, per-bird buttons)
                    action = renderer.handle_event(event)
                    if action:
                        action_type, bird_idx = action
                        if action_type == "cycle_strategy":
                            manager.cycle_strategy(bird_idx)
                        elif action_type == "halve_epsilon":
                            manager.halve_epsilon(bird_idx)
                        elif action_type == "remove":
                            manager.remove_bird(bird_idx)
                            if manager.entries:
                                manager.reset_round()

                    # Check main buttons (only on click)
                    if event.type == pygame.MOUSEBUTTONDOWN and not action:
                        mx, my = event.pos
                        for btn in buttons:
                            if btn["rect"] and btn["rect"].collidepoint(
                                mx - 288, my
                            ):
                                if btn["action"] == "add_bird":
                                    dialog.show()
                                elif btn["action"] == "boost":
                                    manager.boost_all()
                                elif btn["action"] == "evolution":
                                    manager.evolution_enabled = not manager.evolution_enabled
                                elif btn["action"] == "pause":
                                    manager.paused = not manager.paused

        # Sync speed from slider
        manager.speed = renderer.speed

        # Update button labels
        evo_label = "Evolution: ON" if manager.evolution_enabled else "Evolution: OFF"
        buttons[2]["label"] = evo_label
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

        # Draw dialog on top
        if dialog.active:
            dialog.draw(renderer.screen)
            pygame.display.flip()

        renderer.clock.tick(60)

    renderer.quit()


if __name__ == "__main__":
    main()
