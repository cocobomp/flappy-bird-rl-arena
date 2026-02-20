# RL Flappy Bird Race

Projet d'exploration du **reinforcement learning** sur Flappy Bird. Comparez visuellement plusieurs algorithmes RL (Q-Learning, DQN, Double DQN) avec différentes fonctions de reward, le tout en temps réel dans une course multi-oiseaux.

## Apercu

Le mode **Race** fait jouer plusieurs agents RL en simultané dans le meme environnement Flappy Bird. Chaque oiseau est piloté par son propre agent qui apprend en temps réel. Un panneau latéral affiche les stats, les coordonnées du prochain tuyau, et des tooltips explicatifs au survol.

### Algorithmes disponibles

| Algo | Description |
|------|-------------|
| **Q-Learning** | Méthode tabulaire avec discrétisation des états |
| **DQN** | Deep Q-Network avec experience replay et target network |
| **Double DQN** | Variante du DQN qui réduit la surestimation des Q-valeurs |

### Fonctions de reward

| Reward | Description |
|--------|-------------|
| **Basic** | +0.1 en vie, -1.0 à la mort |
| **Distance** | Bonus proportionnel à la proximité du prochain tuyau |
| **Centered** | Bonus pour rester centré dans l'ouverture du tuyau |

## Installation

```bash
# Cloner le projet
git clone <url-du-repo>
cd ReinforcementLearning

# Créer un environnement virtuel
python3.11 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

> **Note** : Python 3.11+ est recommandé. PyTorch (`torch`) est nécessaire pour DQN/Double DQN.

## Lancer le jeu

### Mode Race (recommandé)

Lance une course multi-oiseaux avec interface interactive :

```bash
python race.py
```

**Controles :**
- `Espace` : Pause / Reprendre
- `Fleche haut` : Augmenter la vitesse (x2, x4, x8)
- `Fleche bas` : Diminuer la vitesse
- `Echap` : Quitter
- Clic sur **+ Add Bird** dans le panneau pour ajouter un oiseau avec l'algo et la reward de votre choix

### Mode Entrainement

Entraîne un agent sur Flappy Bird (utilise `flappy-bird-gymnasium`) :

```bash
# Entraînement DQN par défaut (1000 épisodes)
python main.py train --config configs/default.yaml --save

# Entraînement Q-Learning
python main.py train --config configs/q_learning_simple_basic.yaml --save

# Entraînement Double DQN
python main.py train --config configs/double_dqn_simple_distance.yaml --save
```

### Mode Visualisation

Rejoue un agent entraîné dans le jeu Flappy Bird original :

```bash
python main.py play --config configs/dqn_simple_basic.yaml --model models/dqn_simple_basic --episodes 5
```

## Structure du projet

```
ReinforcementLearning/
├── race.py                    # Point d'entrée mode Race (multi-oiseaux)
├── main.py                    # Point d'entrée train/play (agent unique)
├── requirements.txt
├── configs/                   # Configurations YAML des expériences
│   ├── default.yaml
│   ├── dqn_simple_basic.yaml
│   ├── q_learning_simple_basic.yaml
│   └── ...
├── src/
│   ├── agents/                # Algorithmes RL
│   │   ├── base_agent.py      #   ABC commune
│   │   ├── q_learning.py      #   Q-Learning tabulaire
│   │   ├── dqn.py             #   DQN (PyTorch)
│   │   └── double_dqn.py      #   Double DQN
│   ├── environments/          # Wrappers et reward shaping
│   │   ├── wrappers.py        #   SimpleObs (4f), EnrichedObs (7f)
│   │   └── rewards.py         #   Basic, Distance, Centered
│   ├── game/                  # Moteur multi-oiseaux + rendu Pygame
│   │   ├── engine.py          #   Physique Flappy Bird (N oiseaux)
│   │   ├── renderer.py        #   Rendu jeu + panneau + tooltips
│   │   ├── race.py            #   RaceManager (orchestration)
│   │   └── ui.py              #   Dialog d'ajout d'oiseau
│   ├── training/              # Pipeline d'entraînement
│   │   ├── trainer.py         #   Boucle d'entraînement
│   │   ├── config.py          #   Chargement YAML
│   │   └── logger.py          #   Métriques
│   └── visualization/         # Visualisation agent unique
│       ├── renderer.py
│       └── overlay.py
└── tests/                     # 175 tests (pytest)
    ├── test_agents/
    ├── test_environments/
    ├── test_game/
    └── test_training/
```

## Tests

```bash
python -m pytest tests/ -v
```

## Comment ca marche

1. **Moteur de jeu** (`src/game/engine.py`) : Simule N oiseaux en parallèle dans un environnement Flappy Bird avec les memes tuyaux. Physique identique à `flappy-bird-gymnasium`.

2. **Agents RL** : Chaque oiseau a son propre agent qui observe `[position_y, vitesse, distance_tuyau, centre_gap]` et choisit entre "flap" et "ne rien faire" via epsilon-greedy.

3. **Entraînement live** : Les agents apprennent en temps réel pendant la course. L'epsilon decay diminue l'exploration au fil des rounds.

4. **Reward shaping** : Différentes fonctions de reward guident l'apprentissage differemment --- comparez les pour voir laquelle produit les meilleurs résultats.
