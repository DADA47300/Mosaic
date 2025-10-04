import cv2  # Importe la bibliothèque OpenCV pour le traitement d'image.
import numpy as np  # Importe la bibliothèque NumPy pour les opérations numériques.
import sys  # Importe le module sys pour accéder aux arguments de la ligne de commande.
import random  # Importe le module random pour générer des nombres aléatoires.
import glob  # Importe le module glob pour rechercher des chemins de fichiers qui correspondent à un modèle spécifié.
import os  # Importe le module os pour interagir avec le système d'exploitation.

# Définition d'une fonction pour ajuster la couleur d'une image.
def adjust_color(img_src, target_color, strength=1.0):
    img_src = img_src.astype(np.float32)  # Convertit l'image source en type float32 pour les calculs.
    target_color = np.array(target_color, dtype=np.float32)  # Convertit la couleur cible en tableau NumPy de type float32.
    mean_color = img_src.mean(axis=(0, 1))  # Calcule la couleur moyenne de l'image source.
    diff_color = target_color - mean_color  # Calcule la différence entre la couleur cible et la couleur moyenne.
    img_adjusted = img_src + diff_color * strength  # Ajuste l'image en ajoutant la différence pondérée par la force.
    img_adjusted = np.clip(img_adjusted, 0, 255).astype(np.uint8)  # Clampe les valeurs entre 0 et 255 et convertit en uint8.
    return img_adjusted  # Retourne l'image ajustée.

# Définition d'une fonction pour construire une mosaïque d'images.
def build_mosaic(img_src, thumbnails, n):
    height, width, _ = img_src.shape  # Récupère les dimensions de l'image source.
    mosaic = np.zeros((height * n, width * n, 3), dtype=np.uint8)  # Crée une image vide pour la mosaïque.
    spiral_order = spiral_coords(width, height)  # Calcule un ordre en spirale pour placer les vignettes.
    central_thumbnail = None  # Initialise la vignette centrale.
    for idx, (x, y) in enumerate(spiral_order):  # Itère sur les coordonnées en spirale.
        thumb_idx = idx % len(thumbnails)  # Sélectionne une vignette de manière cyclique.
        img_mini = thumbnails[thumb_idx]  # Récupère l'image de la vignette.
        img_mini = cv2.resize(img_mini, (n, n))  # Redimensionne la vignette.
        color = img_src[y, x]  # Récupère la couleur de l'image source à la position (x, y).
        img_adjusted = adjust_color(img_mini, color, 1)  # Ajuste la couleur de la vignette.
        mosaic[y * n:(y + 1) * n, x * n:(x + 1) * n] = img_adjusted  # Place la vignette ajustée dans la mosaïque.
        if x == width // 2 and y == height // 2:
            central_thumbnail = img_adjusted  # Définit la vignette centrale si c'est le centre.
    return mosaic, central_thumbnail  # Retourne la mosaïque et la vignette centrale.

# Définition d'une fonction pour créer une mosaïque d'une image et la sauvegarder.
def mosaic_image(img_src, thumbnails_dir, n, path_output):
    thumbnail_paths = sorted(glob.glob(os.path.join(thumbnails_dir, '*.jpg')))  # Trouve tous les fichiers JPG dans le répertoire.
    thumbnails = [cv2.imread(path) for path in thumbnail_paths if cv2.imread(path) is not None]  # Charge les images des vignettes.
    if not thumbnails:
        print("Error: No thumbnails could be loaded.")  # Affiche une erreur si aucune vignette n'est chargée.
        return

    mosaic, central_thumbnail = build_mosaic(img_src, thumbnails, n)  # Construit la mosaïque.
    cv2.imwrite(path_output, mosaic)  # Sauvegarde la mosaïque dans un fichier.
    print(f"Saved {path_output}")  # Affiche le chemin du fichier sauvegardé.
    return central_thumbnail  # Retourne la vignette centrale.

# Définition d'une fonction pour recadrer et redimensionner une image.
def crop_and_resize(img_src: np.ndarray, crop_rect: tuple, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = crop_rect  # Récupère les coordonnées du rectangle de recadrage.
    w = w if w % 2 == 1 else w - 1  # Assure que la largeur est impaire.
    h = h if h % 2 == 1 else h - 1  # Assure que la hauteur est impaire.
    center_x, center_y = img_src.shape[1] // 2, img_src.shape[0] // 2  # Calcule le centre de l'image source.
    half_width, half_height = w // 2, h // 2  # Calcule la moitié des dimensions du rectangle de recadrage.
    x1, y1 = center_x - half_width, center_y - half_height  # Définit les nouvelles coordonnées du coin supérieur gauche.
    x2, y2 = center_x + half_width, center_y + half_height  # Définit les nouvelles coordonnées du coin inférieur droit.
    img_cropped = img_src[y1:y2, x1:x2]  # Recadre l'image.
    img_resized = cv2.resize(img_cropped, (w, h), interpolation=cv2.INTER_AREA)  # Redimensionne l'image.
    return img_resized  # Retourne l'image recadrée et redimensionnée.

# Définition d'une fonction pour créer une spirale de coordonnées.
def spiral_coords(width, height):
    x, y = width // 2, height // 2  # Commence au centre de l'image.
    dx, dy = 0, -1  # Initialise la direction de la spirale.
    spiral = [(x, y)]  # Commence la spirale avec les coordonnées centrales.
    steps = 1  # Initialise le compteur de pas pour la spirale.
    while len(spiral) < width * height:  # Continue tant que la spirale n'a pas couvert toute l'image.
        for _ in range(2):  # Deux opérations par taille de carré.
            for _ in range(steps):  # Itère sur le nombre de pas dans la direction actuelle.
                x, y = x + dx, y + dy  # Met à jour les coordonnées en fonction de la direction.
                if 0 <= x < width and 0 <= y < height:  # Vérifie que les coordonnées sont dans les limites.
                    spiral.append((x, y))  # Ajoute les nouvelles coordonnées à la spirale.
            dx, dy = -dy, dx  # Tourne à droite.
        steps += 1  # Augmente la taille de la spirale après une boucle complète.

    return spiral  # Retourne les coordonnées de la spirale.

# Définition d'une fonction pour créer une série de mosaïques avec différents niveaux de zoom.
def mosaic_zoom(path_src: str, thumbnails_dir: str, path_prefix_output: str, nw: int, nh: int):
    img_src = cv2.imread(path_src)  # Charge l'image source.
    if img_src is None:
        print(f"Error: Unable to load image at {path_src}")  # Affiche une erreur si l'image ne peut être chargée.
        return

    num_levels = max(nw, nh) - 1  # Détermine le nombre de niveaux de zoom.
    initial_n = 16  # Définit la taille initiale des vignettes.
    increment = (64 - initial_n) / num_levels  # Calcule l'incrément de taille pour chaque niveau de zoom.
    last_thumbnail = None  # Initialise la dernière vignette.

    for i in range(num_levels + 1):  # Itère sur chaque niveau de zoom.
        n = int(initial_n + increment * i)  # Calcule la taille des vignettes pour le niveau actuel.
        if i > 2:
            n = n * 4  # Augmente la taille des vignettes après le troisième niveau.
        # Calcule le rectangle de recadrage pour le niveau de zoom actuel.
        factor = 2 ** i  # Calcule le facteur de zoom.
        cropped_width = img_src.shape[1] // factor  # Calcule la largeur recadrée.
        cropped_height = img_src.shape[0] // factor  # Calcule la hauteur recadrée.
        crop_rect = (img_src.shape[1]//2 - cropped_width//2, img_src.shape[0]//2 - cropped_height//2, img_src.shape[1]//2 + cropped_width//2, img_src.shape[0]//2 + cropped_height//2)  # Définit les coordonnées du rectangle de recadrage.
        resized_img = crop_and_resize(img_src, crop_rect, cropped_width, cropped_height)  # Recadre et redimensionne l'image pour le zoom.
        output_path = f"{path_prefix_output}_{i + 1}.jpg"  # Définit le chemin de sortie pour la mosaïque.
        if i == num_levels:
            last_thumbnail = mosaic_image(resized_img, thumbnails_dir, n, output_path)  # Crée la mosaïque pour le dernier niveau de zoom et la sauvegarde.
        else:
            mosaic_image(resized_img, thumbnails_dir, n, output_path)  # Crée la mosaïque pour les autres niveaux de zoom et la sauvegarde.

    return last_thumbnail  # Retourne la dernière vignette.

# Vérifie si le script est exécuté comme programme principal.
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: make_mosaic.py <SRC_IMG> <THUMBNAILS_DIR> <N_W> <N_H> <OUTPUT_PREFIX>")  # Affiche l'usage correct si les arguments ne sont pas adéquats.
        sys.exit(1)  # Quitte le programme avec un code d'erreur.

    src_img = sys.argv[1]  # Récupère l'image source à partir des arguments de la ligne de commande.
    thumbnails_dir = sys.argv[2]  # Récupère le répertoire des vignettes à partir des arguments.
    n_w = int(sys.argv[3])  # Récupère le nombre de vignettes en largeur à partir des arguments.
    n_h = int(sys.argv[4])  # Récupère le nombre de vignettes en hauteur à partir des arguments.
    output_prefix = sys.argv[5]  # Récupère le préfixe de sortie pour les fichiers à partir des arguments.

    mosaic_zoom(src_img, thumbnails_dir, output_prefix, 8, 8)  # Appelle la fonction mosaic_zoom avec les arguments fournis.
