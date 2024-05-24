import os
import shutil
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
import cv2
from scipy.ndimage import measurements

def regroupement_renommage_datas(chemin_source, chemin_nouveau_dossier, type):
    # Créez le nouveau dossier s'il n'existe pas déjà
    if not os.path.exists(chemin_nouveau_dossier):
        os.makedirs(chemin_nouveau_dossier)

    # Parcourez les sous-dossiers et copiez les images dans le nouveau dossier
    for dossier_racine, _, fichiers in os.walk(chemin_source):
        for fichier in fichiers:
            if fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
                chemin_source_fichier = os.path.join(dossier_racine, fichier)
                # Supprimez la partie spécifiée du nom du fichier
                if type == "images":
                    nouveau_nom = fichier.replace("_leftImg8bit", "")
                elif type == "masks":
                    if "color" in fichier:
                        nouveau_nom = fichier.replace("_gtFine_color", "")
                    else:
                        continue  # Ne pas copier le fichier si le nom ne contient pas "color"
                chemin_destination = os.path.join(chemin_nouveau_dossier, nouveau_nom)
                shutil.copy(chemin_source_fichier, chemin_destination)

                
def normalisation_mask(input_folder, output_folder):
    # Création du dossier de sortie s'il n'existe pas déjà
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Catégories
    cats = {'void': [0, 1, 2, 3, 4, 5, 6],
             'flat': [7, 8, 9, 10],
             'construction': [11, 12, 13, 14, 15, 16],
             'object': [17, 18, 19, 20],
             'nature': [21, 22],
             'sky': [23],
             'human': [24, 25],
             'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}
    
    # Itération sur chaque image du dossier original
    for filename in os.listdir(input_folder):
        # Lecture de l'image
        image_path = os.path.join(input_folder, filename)
        img = img_to_array(load_img(image_path, color_mode = "grayscale"))
        
        # Normalisation des pixels
        labels = np.unique(img)
        img = np.squeeze(img)
        mask = np.zeros((img.shape[0], img.shape[1], 8))
        for label in range(-1, 34):
            if label in cats['void']:
                mask[:,:,0] = np.logical_or(mask[:,:,0], (img==label))
            elif label in cats['flat']:
                mask[:,:,1] = np.logical_or(mask[:,:,1], (img==label))
            elif label in cats['construction']:
                mask[:,:,2] = np.logical_or(mask[:,:,2], (img==label))
            elif label in cats['object']:
                mask[:,:,3] = np.logical_or(mask[:,:,3], (img==label))
            elif label in cats['nature']:
                mask[:,:,4] = np.logical_or(mask[:,:,4], (img==label))
            elif label in cats['sky']:
                mask[:,:,5] = np.logical_or(mask[:,:,5], (img==label))
            elif label in cats['human']:
                mask[:,:,6] = np.logical_or(mask[:,:,6], (img==label))
            elif label in cats['vehicle']:
                mask[:,:,7] = np.logical_or(mask[:,:,7], (img==label))
                
        # Normalisation de chaque couche du masque séparément
        mask_sum = np.clip(mask.sum(axis=2), 0, 7) # Limite la somme à 8
        
        # Trouver la valeur maximale actuelle
        max_val = np.max(mask_sum)
        
        # Normalisation de la somme des valeurs de pixel pour chaque pixel à 8
        if max_val != 0:
            mask_sum = (mask_sum * (7 / max_val)).astype(np.uint8)
        else:
            mask_sum = (mask_sum * 0).astype(np.uint8)  # Si max_val est 0, la somme est déjà 0

        # Définit le nom de fichier de sortie
        file_name = os.path.splitext(filename)[0] + ".png"

        # Enregistrement de l'image
        full_output_file_path = os.path.join(output_folder, file_name)
        img = Image.fromarray(mask_sum.astype(np.uint8), mode='L').save(full_output_file_path)
                
    print("Terminé ! Les masques normalisés ont été enregistrés dans", output_folder)
