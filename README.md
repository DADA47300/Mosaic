# Mosaic / Movie utilities

Ce dépôt contient deux scripts Python simples pour créer des mosaïques d'images et des vidéos de zoom à partir de ces mosaïques.

## Contenu principal
- `make_mosaic.py` : crée une série de mosaïques à partir d'une image source et d'un dossier de miniatures (thumbnails). Pour chaque niveau de zoom, il génère un fichier image avec un préfixe donné (ex. `mosaic_out_1.jpg`, ...).
- `make_movie.py` : génère des vidéos (MP4) en appliquant un zoom progressif sur une série d'images (préfixées) et concatène les vidéos intermédiaires pour obtenir une sortie finale.

## Dépendances
- Python 3.8+ (ou version compatible)
- numpy
- OpenCV (cv2)

Installation (virtualenv conseillé) :

```bash
python -m venv venv_mosaic
source venv_mosaic/bin/activate
pip install --upgrade pip
pip install numpy opencv-python
```

## Utilisation

1) Créer des miniatures

Placez des images miniatures au format `.jpg` dans un dossier (par exemple `thumbnails/`). Le script `make_mosaic.py` recherche les fichiers `*.jpg` dans le dossier que vous lui indiquez.

2) Générer des mosaïques

```bash
python make_mosaic.py <SRC_IMG> <THUMBNAILS_DIR> <N_W> <N_H> <OUTPUT_PREFIX>
# Exemple
python make_mosaic.py source.jpg thumbnails 8 8 mosaic_out
```

Sortie : fichiers `mosaic_out_1.jpg`, `mosaic_out_2.jpg`, ...

3) Générer une vidéo depuis les images mosaïques

```bash
python make_movie.py <MASTER_IMG> <PREFIX_IMG> <DURATION> <ZOOM_MAGNITUDE> <OUTPUT_FILENAME>
# Exemple
python make_movie.py master.jpg mosaic_out 4 4 final_video.mp4
```

Paramètres importants :
- `<DURATION>` : durée en secondes par niveau de zoom
- `<ZOOM_MAGNITUDE>` : nombre de niveaux/images à traiter (doit correspondre au nombre d'images générées par `make_mosaic`)

## Remarques
- `make_mosaic.py` cherche uniquement des fichiers `*.jpg` pour les miniatures. Si vos miniatures sont en `.png`, renommez-les en `.jpg` ou modifiez le script.
- Les tailles de sortie vidéo par défaut dans `make_movie.py` sont grandes (2048×2048) ; réduisez le `target_size` dans le script si votre machine manque de RAM/CPU.
- Si `cv2.imread` retourne `None`, vérifiez le chemin du fichier et ses droits.

## Exemple rapide

```bash
# depuis le répertoire du projet
source venv_mosaic/bin/activate
mkdir -p thumbnails
cp image2.jpg thumbnails/
python make_mosaic.py image1.jpg thumbnails 8 8 mosaic_out
python make_movie.py image1.jpg mosaic_out 3 4 final_video.mp4
```

## Améliorations possibles
- Accepter plusieurs extensions d'image (jpg, png)
- Ajouter argparse pour afficher `--help` et options plus conviviales
- Nettoyer les fichiers temporaires (.mp4 intermédiaires)

---

Si tu veux, je peux :
- ajouter la prise en charge `.png` dans `make_mosaic.py`,
- ajouter un script `run_example.sh` automatisant l'exécution,
- ou convertir les scripts existants pour utiliser `argparse` et afficher de l'aide.
