# RL Flappy Bird — Design Document

## Objectif

Créer une démo visuelle en Python qui compare plusieurs algorithmes de reinforcement learning jouant à Flappy Bird, avec visualisation Pygame en temps réel. Le projet est développé en parallèle par une team d'agents Claude Code.

## Stack technique

- **Langage :** Python 3.11+
- **Environnement :** `flappy-bird-gymnasium` (lib existante)
- **Deep Learning :** PyTorch
- **Visualisation :** Pygame (temps réel)
- **Configuration :** YAML

## Architecture

```
ReinforcementLearning/
├── src/
│   ├── environments/          # Wrappers Gymnasium pour Flappy Bird
│   │   ├── __init__.py
│   │   ├── wrappers.py        # Wrappers d'observation (features, pixels, enrichi)
│   │   └── rewards.py         # Fonctions de reward custom
│   ├── agents/                # Algorithmes RL from scratch (PyTorch)
│   │   ├── __init__.py
│   │   ├── base_agent.py      # Interface commune (ABC)
│   │   ├── q_learning.py      # Q-Learning tabulaire
│   │   ├── dqn.py             # Deep Q-Network
│   │   └── double_dqn.py      # Double DQN
│   ├── training/              # Pipeline d'entraînement
│   │   ├── __init__.py
│   │   ├── trainer.py         # Boucle d'entraînement générique
│   │   ├── config.py          # Configurations d'expériences
│   │   └── logger.py          # Logging des métriques
│   └── visualization/         # Rendu Pygame + overlay
│       ├── __init__.py
│       ├── renderer.py        # Rendu du jeu en temps réel
│       └── overlay.py         # Graphiques de métriques en overlay
├── configs/                   # Fichiers YAML de config d'expériences
├── models/                    # Modèles sauvegardés
├── main.py                    # Point d'entrée
├── requirements.txt
└── README.md
```

## Algorithmes RL

| Algorithme | Type | Description |
|---|---|---|
| Q-Learning | Tabulaire | Baseline. Table Q discrétisée sur features simples. Montre les limites du tabulaire. |
| DQN | Deep, value-based | Réseau de neurones, experience replay buffer, target network avec soft update. |
| Double DQN | Deep, value-based | Idem DQN mais découple sélection et évaluation pour corriger la surestimation des Q-values. |

Tous les agents implémentent `BaseAgent` (ABC) avec : `select_action(state)`, `train_step(batch)`, `save(path)`, `load(path)`.

## Variations

### Reward shaping
- **Basique :** +1 par step de survie, -1000 à la mort
- **Distance :** reward proportionnelle à la distance horizontale parcourue
- **Optimal :** bonus quand l'oiseau est centré verticalement entre les tuyaux

### Espaces d'observation
- **Simple :** `(y_bird, velocity, dist_next_pipe, y_gap)` — 4 features
- **Enrichi :** + `(dist_2nd_pipe, y_gap_2nd, delta_y_to_gap)` — 7 features
- **Pixels :** image 84x84 grayscale, stack de 4 frames — pour DQN avec CNN

### Hyperparamètres
- Learning rate : [1e-4, 5e-4, 1e-3]
- Epsilon decay : linéaire vs exponentiel
- Discount factor (gamma) : [0.95, 0.99]
- Batch size : [32, 64]
- Architecture réseau : [64-64, 128-128, 256-128]

## Visualisation Pygame

- Fenêtre principale : le jeu Flappy Bird en temps réel avec l'agent jouant
- Overlay en haut : nom de l'algo, épisode courant, score actuel, meilleur score
- Menu de sélection : choisir l'algo et la configuration avant lancement
- Switch entre agents pendant le replay

## Team d'agents Claude

| Agent | Responsabilité |
|---|---|
| Agent Environnement | Setup projet, `requirements.txt`, wrappers Gymnasium, fonctions de reward |
| Agent Algorithmes | Implémentation Q-Learning, DQN, Double DQN avec PyTorch |
| Agent Training | Pipeline d'entraînement, configs YAML, logging des métriques |
| Agent Visualisation | Rendu Pygame temps réel, overlay métriques, menu de sélection |

Les agents travaillent en parallèle sur des modules indépendants, avec des interfaces définies (BaseAgent, config schema, métriques format).
