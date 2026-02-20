# Flappy Bird RL Arena

Projet pedagogique de **reinforcement learning** applique a Flappy Bird. Plusieurs agents RL (Q-Learning, DQN, Double DQN) s'affrontent en temps reel dans une course multi-oiseaux, avec des parametres entierement configurables.

## Demo

Lance une course multi-oiseaux avec interface interactive :

```bash
python race.py
```

3 oiseaux DQN demarrent avec des strategies d'exploration differentes. Ajoutez-en d'autres via le dialog, modifiez les parametres en live, et observez l'apprentissage.

## Fonctionnalites

### Algorithmes RL

| Algo | Description |
|------|-------------|
| **Q-Learning** | Methode tabulaire avec discretisation des etats |
| **DQN** | Deep Q-Network avec experience replay et target network |
| **Double DQN** | Variante du DQN qui reduit la surestimation des Q-valeurs |

### Fonctions de reward

| Reward | Description |
|--------|-------------|
| **Basic** | +0.1 en vie, -1.0 a la mort |
| **Distance** | Bonus proportionnel a la proximite du prochain tuyau |
| **Centered** | Bonus pour rester centre dans l'ouverture du tuyau |
| **Smart** | Centrage + direction + progression + penalite mort configurable |

### Strategies d'exploration

| Strategie | Description |
|-----------|-------------|
| **Random** | 50/50 aleatoire (demo, peu efficace) |
| **Gravity** | 12% de flap — compense l'asymetrie flap(-9) vs gravite(+1) |
| **Heuristic** | Controleur PD : tient compte de la position ET de la velocite |
| **Guided** | Gravity loin du tuyau + Heuristic pres (recommande) |

### Parametres configurables

Chaque oiseau peut etre configure independamment via le dialog :

- **Apprentissage** : epsilon initial, learning rate, epsilon decay
- **Recompenses** : penalite de mort, bonus par tuyau, reward de survie
- **Strategie** : seuil de flap, bruit aleatoire

### Interface interactive

- Panneau lateral avec stats en temps reel par oiseau
- Q-values affichees (visualisation de l'apprentissage)
- Badge EXPLORE / APPREND (exploration vs exploitation)
- Boutons par oiseau : changer strategie, diviser epsilon, supprimer
- Slider de vitesse (x1 a x64)
- Panneau pedagogique expliquant les concepts RL
- Tooltips au survol sur les algorithmes et rewards

## Controles

| Touche | Action |
|--------|--------|
| `Espace` | Pause / Reprendre |
| `Fleche haut` | Vitesse x2 |
| `Fleche bas` | Vitesse /2 |
| `B` | Boost : epsilon = 0.05 pour tous |
| `Echap` | Quitter |

## Installation

```bash
git clone https://github.com/cocobomp/flappy-bird-rl-arena.git
cd flappy-bird-rl-arena

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Python 3.11+ recommande. PyTorch necessaire pour DQN / Double DQN.

## Mode entrainement (agent unique)

```bash
python main.py train --config configs/default.yaml --save
python main.py play --config configs/dqn_simple_basic.yaml --model models/dqn_simple_basic --episodes 5
```

## Structure

```
├── race.py                     # Mode Race (multi-oiseaux)
├── main.py                     # Mode train/play (agent unique)
├── requirements.txt
├── configs/                    # Configurations YAML
├── src/
│   ├── agents/                 # Q-Learning, DQN, Double DQN
│   ├── environments/           # Wrappers + reward shaping
│   ├── game/                   # Moteur multi-oiseaux + UI Pygame
│   │   ├── engine.py           #   Physique (N oiseaux, tuyaux partages)
│   │   ├── renderer.py         #   Rendu + panels + tooltips
│   │   ├── race.py             #   Orchestration agents/oiseaux
│   │   ├── strategies.py       #   Strategies d'exploration
│   │   └── ui.py               #   Dialog d'ajout configurable
│   ├── training/               # Pipeline d'entrainement
│   └── visualization/          # Rendu agent unique
└── tests/                      # 175 tests
```

## Tests

```bash
python -m pytest tests/ -v
```

## Comment ca marche

1. **Moteur** : Simule N oiseaux en parallele avec physique identique a `flappy-bird-gymnasium`
2. **Observation** : `[position_y, vitesse, distance_tuyau, centre_gap]` — 4 valeurs normalisees
3. **Decision** : Epsilon-greedy — explore avec la strategie choisie ou exploite le reseau entraine
4. **Apprentissage** : Les agents apprennent en temps reel pendant la course
5. **Reward shaping** : Differentes fonctions guident l'apprentissage — Smart + Guided donne les meilleurs resultats
