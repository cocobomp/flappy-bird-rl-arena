# Multi-Bird RL Race — Design Document

## Objectif

Interface Pygame montrant plusieurs oiseaux RL jouant simultanement dans le meme environnement Flappy Bird, avec panneau de controle lateral pour ajouter/configurer des oiseaux et observer l'apprentissage en direct.

## Concept

- Un seul jeu Flappy Bird (memes tuyaux pour tous)
- Chaque oiseau = un agent RL independant avec sa couleur, son algo, et sa reward
- Entrainement live : on voit les oiseaux apprendre en temps reel
- Round = tant qu'au moins 1 oiseau est vivant, puis reset pour tous
- Les agents continuent d'apprendre entre les rounds

## Layout

```
+-----------------------------------+----------------------+
|                                   |  ROUND 47            |
|   [R](DQN)          |   |        |                      |
|      [B](DDQN)      | G |        |  [R] DQN Basic       |
|                      |   |        |      Score: 3        |
|              [G](QL)              |      Best: 12        |
|                                   |                      |
|         FLAPPY BIRD               |  [B] DDQN Distance   |
|                                   |      Score: 5 (best) |
|                                   |      Best: 18        |
|                                   |                      |
|                                   |  [G] QL Basic        |
|                                   |      Score: 0 (dead) |
|                                   |      Best: 2         |
|                                   |                      |
|                                   |  [+ Add bird]        |
|                                   |  Speed: x1 x2 x5    |
|                                   |  [Pause]             |
+-----------------------------------+----------------------+
```

## Environnement custom multi-oiseaux

Puisque flappy-bird-gymnasium gere un seul oiseau, on cree un environnement custom :

- Physique Flappy Bird : gravite, flap, collisions avec tuyaux/sol/plafond
- Genere les tuyaux (position, gap) de facon procedurale
- Gere N oiseaux avec chacun son propre etat (y, velocity, alive)
- Observation par oiseau : (player_y, velocity, dist_next_pipe, gap_center) — 4 features, mode simple
- Actions : 0=rien, 1=flap — par oiseau
- Rendu Pygame custom (pas via gymnasium render)

## Panneau de controle

- Liste des oiseaux : couleur, nom (algo+reward), score, best score, statut
- Bouton "Add bird" : selection algo (Q-Learning/DQN/DDQN) + reward (Basic/Distance/Centered)
- Controle vitesse : x1, x2, x5, x10
- Pause/Play

## Stack

- Python 3.11, Pygame, PyTorch, NumPy
- Reutilise les agents existants (QLearningAgent, DQNAgent, DoubleDQNAgent)
- Reutilise les reward functions existantes (BasicReward, DistanceReward, CenteredReward)
- Nouvel environnement custom (pas gymnasium, juste la physique)
