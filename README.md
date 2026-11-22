# Champions Celebration AR Mirror - Reconnaissance de Poses & Réalité Augmentée en Temps Réel

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe%20Pose-orange.svg)](https://developers.google.com/mediapipe)
[![Audio Engine](https://img.shields.io/badge/Pygame-Audio-red.svg)](https://www.pygame.org/)
[![Cloud Backend](https://img.shields.io/badge/Supabase-Storage%20%26%20DB-emerald.svg)](https://supabase.com/)
[![Status](https://img.shields.io/badge/Status-In%20Active%20Development-yellow.svg)]()

> **Application interactive de Vision par Ordinateur et Réalité Augmentée (AR)** transformant une simple webcam en miroir interactif : détection en temps réel des poses et célébrations iconiques de stars du football mondial, incrustation dynamique des joueurs en réalité augmentée, segmentation d'arrière-plan, effets sonores immersifs et synchronisation Cloud.

---

## 📌 1. Idée Générale & Vision du Projet

### La Vision
Créer une expérience de **Miroir Augmenté (AR Mirror)** interactive et ludique destinée aux fans de sport, aux événements interactifs ou aux installations immersives : l'utilisateur reproduit devant sa webcam la célébration d'un joueur célèbre (le « Siuuu » de Ronaldo, les bras ouverts de Bellingham, le doigt sur la tempe de Rashford, etc.), et le système reconnaît instantanément la posture, valide le geste grâce à un temps de maintien, et l'intègre virtuellement aux côtés de son idole dans un stade en liesse avec sons de foule et capture automatique de clichés.

### Les Défis Techniques
1. **Traitement en temps réel sans latence perceptible** (30 FPS constant) sur CPU grand public.
2. **Robustesse de la reconnaissance géométrique** : invariance à la distance de la caméra, à la corpulence de l'utilisateur et aux décalages angulaires.
3. **Élimination des faux positifs** : différencier un geste intentionnel d'un mouvement anodin grâce à une machine à états à temporisation d'accumulation (*dwell time*) et hystérésis (*debounce*).
4. **Compositing vidéo fluide** : segmentation de l'humain, incrustation douce avec masque de contour dégradé (*feathering*) et rendu double écran côte à côte (*Split-Canvas Dual Pane*).

---

## 🏗️ 2. Architecture Globale du Système

L'architecture est structurée en un pipeline modulaire synchrone optimisé pour le temps réel :

```
                        [ Flux Caméra (Webcam 640x480) ]
                                      │
                                      ▼
                 [ Détection Squelette (MediaPipe Pose 33 pts) ]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   [ Moteur Géométrique de Célébrations ]        [ Module de Segmentation Humaine ]
   (Normalisation échelle & ratios corps)       (Inpainting / Masque flou / Détourage)
                 │                                         │
                 ▼                                         │
   [ Machine à États Finis (FSM) ]                         │
   - Dwell Charge (0.8s hold)                              │
   - Debounce Grace (0.4s)                                 │
   - Anti-répétition d'assets (Buffer FIFO)                │
                 │                                         │
                 ▼                                         ▼
   [ Audio Engine (Pygame) ]                     [ Moteur de Rendu AR (Dual-Pane) ]
   (Effets sonores / Cris de stade)             - Gauche : Live Feed + Squelette HUD
                 │                              - Droite : Stade Virtuel + Incrustation Joueur
                 └────────────────────┬────────────────────┘
                                      ▼
                       [ Événements & Persistance ]
                       - Déclencheur Screenshot auto
                       - Galerie interactive (CustomTkinter)
                       - Synchronisation Cloud (Supabase Storage/DB)
```

---

## 3. Choix Techniques & Justifications

### 1. MediaPipe Pose Landmarker (Modèle Task Lite) vs Détection d'Image Brute (CNN / YOLO)
- **Choix retenu** : `MediaPipe Pose Landmarker Lite (33 repères 3D)` via le fichier de modèle Google TFLite embarqué (`pose_landmarker_lite.task`).
- **Pourquoi ?**
  - **Latence ultra-faible** : Fonctionne à >30 FPS sur simple CPU sans nécessiter de carte graphique dédiée.
  - **Indépendance visuelle** : L'analyse repose sur la géométrie squelettique (angles, coordonnées normalisées, distances relatives), insensible aux variations de vêtements, d'éclairage ou de fond.
  - **Données 3D natives** : Coordonnées relatives (x, y, z) permettant d'évaluer la profondeur des bras et des articulations.

### 2. Algorithmes Heuristiques Géométriques & Métriques Normalisées
Au lieu d'un modèle opaque difficile à corriger, la détection repose sur des **ratios anatomiques normalisés** :
- **Largeur des épaules** (`shoulder_width`) et **hauteur du tronc** (`torso_height`) utilisées comme unités de mesure locales.
- Tolérances adaptatives pour chaque célébration :
  - *Jude Bellingham* : Envergure des bras > 1.8 × la largeur d'épaule, poignets alignés au niveau des épaules.
  - *Lionel Messi* : Deux poignets élevés simultanément au-dessus des oreilles et du nez.
  - *Marcus Rashford* : Index touchant la tempe (distance index-oreille < 0.35 × tronc), coude plié, bras opposé au repos.
  - *Kylian Mbappé* : Bras croisés sur la poitrine, poignets sous les aisselles opposées.
  - *Erling Haaland* : Posture du lotus / méditation zen, mains jointes au sternum, coudes écartés.
  - *Cristiano Ronaldo* : Posture d'atterrissage du "Siuuu", poignets balayés vers le bas et l'arrière, sous la ligne des hanches.

### 3. Machine à États Finis (FSM) avec Dwell Time et Debounce
- **Dwell Timer (0.8 s)** : L'utilisateur doit maintenir la pose stabilisée pendant 800 ms pour déclencher l'événement, évitant tout déclenchement parasite lors de mouvements normaux.
- **Debounce Grace (0.4 s)** : Si l'utilisateur perd brièvement la pose (occlusion d'une main, mouvement rapide), un temps de grâce évite la coupure brutale de l'animation.
- **Lissage Alpha (0.0 -> 1.0)** : Fondu enchaîné visuel pour une apparition progressive du joueur à côté de l'utilisateur.
- **Gestionnaire d'Historique FIFO** : Mémoire tampon des deux derniers assets affichés pour garantir la variété visuelle et éviter la répétition d'un même joueur/pose d'affilée.

### 4. Rendu Visuel Dual-Pane & Segmentation
- **Split-Screen Canvas (1280 x 480)** :
  - **Panneau gauche (640 x 480)** : Flux miroir brut avec tracé squelettique, boîte englobante et jauge circulaire/barre de chargement du *dwell time*.
  - **Panneau droit (640 x 480)** : Vue stade avec détourage en direct de l'utilisateur par inpainting de fond (`BackgroundInpainter`), incrustation en alpha-blending du joueur détouré (`PlayerOverlayManager`), et effets graphiques d'impact.

---

## 4. État d'Avancement & Feuille de Route (Roadmap)

Le projet est conçu de façon modulaire avec des jalons clairs :

| Composant | Statut | Description Technique |
| :--- | :---: | :--- |
| **Pipeline Vidéo & Caméra** | ✅ Implémenté | Capture webcam multithreadée/optimisée OpenCV avec mode miroir |
| **Extraction de Squelette** | ✅ Implémenté | Intégration MediaPipe Pose Landmarker Lite avec inférence 33 points 3D |
| **Moteur Heuristique (6 Célébrations)** | ✅ Implémenté | Algorithmes pour Bellingham, Messi, Rashford, Mbappé, Haaland, Ronaldo |
| **FSM Dwell Time & Debounce** | ✅ Implémenté | Gestion du temps de maintien, lissage alpha et historique anti-répétition |
| **Suite de Tests Unitaires** | ✅ Implémenté | `test_pipeline.py` validant la FSM, la détection et les overlays |
| **Player Overlay & Blending** | ✅ Implémenté | Calibrage de positionnement du joueur virtuel par rapport à la bounding box |
| **Background Inpainting & Masque** | 🔄 Partiel | Calibration du fond (touche 'C') et masque de segmentation à parfaire |
| **Audio Engine (Pygame)** | 🔄 Intégration | Déclenchement des sons de célébration et d'ambiance de stade |
| **Galerie UI (CustomTkinter)** | ⏳ À implémenter | Interface de visualisation des clichés pris lors des célébrations |
| **Cloud Sync (Supabase)** | ⏳ À finaliser | Upload automatique des screenshots et métadonnées sur Supabase Storage/DB |
| **Classification Hybride ML / DL** | ⏳ Futur | Entraînement d'un classifieur SVM/MLP en complément des règles géométriques |

---

## 📁 5. Arborescence du Projet

```text
handsign_recognizing/
├── assets/
│   ├── audio/                  # Fichiers audio des célébrations (.wav / .mp3)
│   ├── models/                 # Modèle MediaPipe pose_landmarker_lite.task
│   └── players/                # Images PNG détourées des stars du football
├── screenshots/                # Captures d'écran sauvegardées localement
├── src/
│   ├── __init__.py
│   ├── camera.py               # Gestion du flux vidéo OpenCV & calibration
│   ├── celebrations.py         # Moteur heuristique d'évaluation des célébrations
│   ├── config.py               # Constantes, seuils géométriques, couleurs, Supabase
│   ├── player_overlay.py       # Positionnement et incrustation alpha des joueurs
│   ├── pose_detector.py        # Wrapper MediaPipe Pose Landmarker
│   ├── renderer.py             # Rendu graphique Canvas Dual-Pane et HUD
│   ├── segmenter.py            # Segmentation d'arrière-plan et inpainting
│   └── state_machine.py        # FSM avec Dwell Time, Debounce et gestion d'assets
├── main.py                     # Boucle principale de l'application interactive
├── test_pipeline.py            # Tests de validation du pipeline et de la FSM
├── requirements.txt            # Dépendances Python du projet
└── README.md                   # Documentation technique détaillée
```

---

## 6. Installation & Utilisation

### 1. Prérequis
- Système d'exploitation : Ubuntu (WSL2 ou natif) ou Linux
- Python 3.10+
- Webcam active et accessible

### 2. Installation de l'environnement virtuel
```bash
# Se placer dans le répertoire du projet

# Créer l'environnement virtuel (s'il n'existe pas déjà)
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Exécution des tests de validation
```bash
python3 test_pipeline.py
```

### 4. Lancement de l'application
```bash
python3 main.py
```

### 5. Contrôles Interactifs
- **[C]** : **Calibrer la pièce** (s'écarter 2 secondes pour capturer le fond sans utilisateur).
- **[S]** : **Prendre un Screenshot** manuel à tout moment.
- **[M]** : **Activer / Couper le son**.
- **[Q]** : **Quitter l'application**.

---

## 🔮 7. Perspectives & Évolutions

1. **Reconnaissance multi-utilisateurs simultanée** : Détecter et célébrer à plusieurs devant la caméra (mode duel ou duo).
2. **Classifieur hybride statistique** : Coupler les règles géométriques actuelles avec un classifieur de Machine Learning entraîné sur des séries temporelles (ex: LSTM pour capturer le mouvement dynamique de la course de célébration).
3. **Filtres vidéo avancés** : Effets de confettis, flashs de stade et éclairages dynamiques en réalité augmentée.
