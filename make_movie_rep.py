import cv2  # Importe la bibliothèque OpenCV pour le traitement d'image.
import numpy as np  # Importe la bibliothèque NumPy pour les opérations numériques.
import sys  # Importe le module sys pour accéder aux arguments de la ligne de commande.
import glob  # Importe le module glob pour rechercher des chemins de fichiers qui correspondent à un modèle spécifié.
import os  # Importe le module os pour interagir avec le système d'exploitation.
from make_mosaic import mosaic_zoom  # Importe la fonction mosaic_zoom du module make_mosaic pour générer des mosaïques.

# Définit une fonction pour créer une série linéaire de valeurs entre deux nombres.
def linear_range(val1: int, val2: int, n: int):
    return np.linspace(val1, val2, n).astype(int).tolist()  # Utilise linspace pour générer n valeurs linéaires entre val1 et val2, convertit en int et retourne comme liste.

# Définit une fonction pour interpoler entre deux rectangles sur n étapes.
def rect_morphing(rect_src: tuple, rect_dest: tuple, nb_steps: int):
    x1_values = linear_range(rect_src[0], rect_dest[0], nb_steps)  # Génère les valeurs interpolées pour le premier coin x.
    y1_values = linear_range(rect_src[1], rect_dest[1], nb_steps)  # Génère les valeurs interpolées pour le premier coin y.
    x2_values = linear_range(rect_src[2], rect_dest[2], nb_steps)  # Génère les valeurs interpolées pour le second coin x.
    y2_values = linear_range(rect_src[3], rect_dest[3], nb_steps)  # Génère les valeurs interpolées pour le second coin y.
    
    rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(x1_values, y1_values, x2_values, y2_values)]  # Crée une liste de tuples représentant les rectangles interpolés.
    return rects  # Retourne la liste des rectangles.

# Définit une fonction pour créer un film à partir d'une image et d'une série de transformations rectangulaires.
def makemovie(img: np.ndarray, rect_src: tuple, rect_dest: tuple, output_filename: str, w: int, h: int, fps: int, duration: int):
    total_frames = fps * duration  # Calcule le nombre total de frames basé sur la durée et les fps.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Définit le codec vidéo.
    video = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))  # Crée un objet VideoWriter pour écrire le fichier vidéo.

    intermediate_rects = rect_morphing(rect_src, rect_dest, total_frames)  # Obtient les rectangles interpolés pour l'animation.
    
    for rect in intermediate_rects:  # Boucle sur chaque rectangle interpolé.
        x1, y1, x2, y2 = rect  # Décompose le rectangle.
        roi = img[y1:y2, x1:x2]  # Extrai la région d'intérêt de l'image.
        resized_roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)  # Redimensionne la région d'intérêt à la taille désirée.
        video.write(resized_roi)  # Écrit la frame redimensionnée dans le fichier vidéo.

    video.release()  # Libère le VideoWriter.
    print(f"Video saved as {output_filename}")  # Affiche le nom du fichier vidéo sauvegardé.

# Définit une fonction pour calculer l'échelle de redimensionnement entre deux images.
def compute_scale(img, next_img, index):
    ratio_width = img.shape[1] / next_img.shape[1]  # Calcule le ratio de largeur.
    ratio_height = img.shape[0] / next_img.shape[0]  # Calcule le ratio de hauteur.
    max_ratio = max(ratio_width, ratio_height)  # Prend le maximum des deux ratios.
    
    if index >= 6:  # Si l'index est 6 ou plus,
        default_scale = 1.75  # Utilise une échelle par défaut plus petite.
    else:
        default_scale = 2  # Sinon, utilise une échelle par défaut de 2.
    
    if max_ratio < 1.7:  # Si le ratio maximal est inférieur à 1.7,
        max_scale = default_scale  # Utilise l'échelle par défaut.
    elif max_ratio > 2:  # Si le ratio maximal est supérieur à 2,
        max_scale = 2.35  # Utilise une échelle maximale de 2.35.
    else:
        max_scale = default_scale  # Sinon, utilise l'échelle par défaut.

    print("max_scale calculated:", max_scale)  # Affiche l'échelle maximale calculée.
    return max_scale  # Retourne l'échelle maximale.

# Définit une fonction pour redimensionner une image et écrire les frames pour une vidéo de zoom.
def resize_and_write_frames(img, max_scale, output_video_path, duration, fps, target_size):
    img = cv2.resize(img, target_size)  # Redimensionne l'image à la taille cible.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Définit le codec vidéo.
    video = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)  # Crée un objet VideoWriter.
    center = (target_size[0] // 2, target_size[1] // 2)  # Définit le centre pour la transformation de rotation.
    
    base_scale = np.linspace(1, max_scale, int(duration * fps))  # Crée une série linéaire d'échelles pour le zoom.
    for scale in base_scale:  # Boucle sur chaque échelle de zoom.
        M = cv2.getRotationMatrix2D(center, 0, scale)  # Crée une matrice de rotation sans rotation, avec zoom.
        frame = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)  # Applique la transformation à l'image.
        video.write(frame)  # Écrit la frame dans le fichier vidéo.
    video.release()  # Libère le VideoWriter.

# Définit une fonction pour lire et ajouter des frames d'une vidéo existante à un VideoWriter.
def read_and_append_video(current_video_path, video_writer):
    cap = cv2.VideoCapture(current_video_path)  # Ouvre le fichier vidéo.
    if not cap.isOpened():  # Si le fichier ne peut pas être ouvert,
        print(f"Failed to open {current_video_path}")  # Affiche un message d'erreur.
        return False

    while True:
        ret, frame = cap.read()  # Lit une frame de la vidéo.
        if not ret:  # Si aucune frame n'est lue,
            break  # Sort de la boucle.
        video_writer.write(frame)  # Ajoute la frame au VideoWriter.
    cap.release()  # Libère le capture vidéo.
    return True  # Retourne True si le processus est réussi.

# Définit une fonction pour effectuer un zoom x2 et créer une vidéo mp4.
def zoom_x2_mp4(image_path, output_video_path, next_image_path, duration, index, fps=24, target_size=(2048, 2048), is_last=False):
    img = cv2.imread(image_path)  # Charge l'image à partir du chemin spécifié.
    if img is None:  # Si l'image ne peut être chargée,
        raise FileNotFoundError(f"Cannot find the file: {image_path}")  # Lève une exception de fichier non trouvé.

    max_scale = 5 if is_last else compute_scale(img, cv2.imread(next_image_path), index)  # Calcule l'échelle maximale, avec une échelle spéciale si c'est la dernière image.
    resize_and_write_frames(img, max_scale, output_video_path, duration, fps, target_size)  # Redimensionne l'image et écrit les frames pour la vidéo.

# Définit une fonction récursive pour créer une série de vidéos zoomées.
def recursive_zoom_mp4(input_prefix, i_max, final_video_path, duration, fps=24, video_writer=None, target_size=(2048, 2048), i=1):
    if i > i_max:  # Si l'index est plus grand que le maximum,
        if video_writer:  # Et si un VideoWriter est présent,
            print(f"{final_video_path} completed")  # Affiche un message indiquant la complétion.
            video_writer.release()  # Libère le VideoWriter.
        return

    image_path = f"{input_prefix}_{i}.jpg"  # Définit le chemin de l'image actuelle.
    next_image_path = f"{input_prefix}_{i+1}.jpg" if i < i_max else None  # Définit le chemin de la prochaine image si ce n'est pas la dernière.
    current_video_path = f"{input_prefix}_{i}.mp4"  # Définit le chemin de la vidéo actuelle.
    
    zoom_x2_mp4(image_path, current_video_path, next_image_path, duration, i, fps, target_size, is_last=(i == i_max))  # Crée une vidéo zoomée.
    
    if video_writer is None:  # Si aucun VideoWriter n'est présent,
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Définit le codec vidéo.
        video_writer = cv2.VideoWriter(final_video_path, fourcc, fps, target_size)  # Crée un VideoWriter pour le fichier final.

    if not read_and_append_video(current_video_path, video_writer):  # Lit et ajoute les frames de la vidéo actuelle au VideoWriter.
        return

    recursive_zoom_mp4(input_prefix, i_max, final_video_path, duration, fps, video_writer, target_size, i + 1)  # Appelle récursivement la fonction pour la prochaine image.

# Définit une fonction pour effectuer un zoom infini à partir d'une image maîtresse, en utilisant des vignettes pour générer des mosaïques à différents niveaux de zoom.
def infinite_zoom(master_img, thumbnails_dir, output_filename, duration, zoom_levels=2, fps=24, nw=8, nh=8):
    video_filenames = []  # Liste pour stocker les noms de fichiers vidéo générés.
    for i in range(zoom_levels):  # Boucle sur chaque niveau de zoom.
        if i == 0:  # Si c'est le premier niveau,
            central_thumbnail = mosaic_zoom(master_img, thumbnails_dir, "mosaic_image", nw, nh)  # Génère la mosaïque initiale.
            if central_thumbnail is None:  # Si aucune vignette centrale n'est obtenue,
                print("Error generating mosaic.")  # Affiche un message d'erreur.
                return
            central_thumbnail_path = f"central_thumbnail_{i}.jpg"  # Définit le chemin de la vignette centrale.
            cv2.imwrite(central_thumbnail_path, central_thumbnail)  # Sauvegarde la vignette centrale.
            video_filename = f"{i}_{output_filename}"  # Définit le nom du fichier vidéo.
            recursive_zoom_mp4("mosaic_image", 8, video_filename, duration)  # Crée une série de vidéos zoomées.
            video_filenames.append(video_filename)  # Ajoute le nom du fichier vidéo à la liste.
        else:
            central_thumbnail_path = f"central_thumbnail_{i-1}.jpg"  # Utilise la vignette centrale précédente pour le prochain niveau.
            central_thumbnail = mosaic_zoom(central_thumbnail_path, thumbnails_dir, f"{i}_mosaic_image", 6, 6)  # Génère la prochaine mosaïque.
            if central_thumbnail is None:  # Si aucune vignette centrale n'est obtenue,
                print(f"Error generating mosaic at level {i}.")  # Affiche un message d'erreur.
                return
            central_thumbnail_path = f"central_thumbnail_{i}.jpg"  # Met à jour le chemin de la vignette centrale.
            cv2.imwrite(central_thumbnail_path, central_thumbnail)  # Sauvegarde la nouvelle vignette centrale.
            video_filename = f"{i}_{output_filename}"  # Définit le nom du nouveau fichier vidéo.
            recursive_zoom_mp4(f"{i}_mosaic_image", 6, video_filename, duration)  # Crée une nouvelle série de vidéos zoomées.
            video_filenames.append(video_filename)  # Ajoute le nom du fichier vidéo à la liste.

    # Concatène les vidéos
    concatenate_videos(video_filenames, output_filename)  # Appelle la fonction pour concaténer les vidéos en un seul fichier.

# Définit une fonction pour concaténer plusieurs fichiers vidéo en un seul.
def concatenate_videos(video_filenames, final_output):
    if os.path.exists(final_output):  # Si le fichier de sortie final existe déjà,
        os.remove(final_output)  # Le supprime pour éviter les conflits.
    temp_output = "temp_" + final_output  # Crée un nom de fichier temporaire pour éviter de réécrire le fichier final pendant le traitement.
    cap = cv2.VideoCapture(video_filenames[0])  # Ouvre le premier fichier vidéo.
    if not cap.isOpened():  # Si le fichier ne peut pas être ouvert,
        print("Error opening the first video file.")  # Affiche un message d'erreur.
        return
    ret, frame = cap.read()  # Lit la première frame de la vidéo.
    if not ret:  # Si la frame ne peut être lue,
        print("Error reading the first frame of the video.")  # Affiche un message d'erreur.
        cap.release()  # Libère le capteur vidéo.
        return
    height, width, layers = frame.shape  # Récupère les dimensions de la frame.
    cap.release()  # Libère le capteur vidéo après la lecture de la première frame.

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Définit le codec vidéo.
    video = cv2.VideoWriter(temp_output, fourcc, 24, (width, height))  # Crée un VideoWriter pour écrire le fichier vidéo temporaire.

    for filename in video_filenames:  # Boucle sur chaque nom de fichier vidéo à concaténer.
        cap = cv2.VideoCapture(filename)  # Ouvre le fichier vidéo.
        while True:
            ret, frame = cap.read()  # Lit les frames de la vidéo.
            if not ret:  # Si aucune frame n'est lue,
                break  # Sort de la boucle.
            video.write(frame)  # Écrit la frame dans le fichier vidéo temporaire.
        cap.release()  # Libère le capteur vidéo après avoir terminé de lire toutes les frames.

    video.release()  # Libère le VideoWriter après avoir terminé d'écrire toutes les frames.

    os.rename(temp_output, final_output)  # Renomme le fichier temporaire en fichier final.
    print("Videos concatenated successfully into", final_output)  # Affiche un message confirmant la réussite de la concaténation.

# Bloc principal qui vérifie si le script est exécuté directement et gère les arguments de ligne de commande.
if __name__ == "__main__":
    if len(sys.argv) != 6:  # Si le nombre d'arguments n'est pas égal à 6,
        print("Usage: make_movie_rep.py <MASTER_IMG> <THUMBNAILS_DIR> <DURATION> <ZOOM_MAGNITUDE> <OUTPUT_FILENAME>")  # Affiche les instructions d'utilisation.
        sys.exit(1)  # Quitte le programme avec un code d'erreur.

    master_img = sys.argv[1]  # Récupère l'image maîtresse à partir des arguments de la ligne de commande.
    thumbnails_dir = sys.argv[2]  # Récupère le répertoire des vignettes à partir des arguments.
    duration = int(sys.argv[3])  # Récupère la durée à partir des arguments.
    zoom_magnitude = int(sys.argv[4])  # Récupère la magnitude du zoom à partir des arguments.
    output_filename = sys.argv[5]  # Récupère le nom du fichier de sortie à partir des arguments.

    infinite_zoom(master_img, thumbnails_dir, output_filename, duration)  # Appelle la fonction infinite_zoom avec les arguments fournis.
