# -*- coding: utf-8 -*-
# 1. Library imports
import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
import tensorflow as tf
# Les functions écrites dans un fichier externe
from functions import *
from keras_segmentation.models.unet import vgg_unet

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

#def label_to_color_image(label):
#    label_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Définir les couleurs des classes
#    colored_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
#    for i in range(len(label_colors)):
#        colored_label[label == i] = label_colors[i]
#    return colored_label

couleur = [
    [0, 0, 255],  # Classe 0: Bleu
    [0, 255, 0],  # Classe 1: Vert
    [255, 0, 0],  # Classe 2: Rouge
    [255, 255, 0],  # Classe 3: Jaune
    [255, 0, 255],  # Classe 4: Magenta
    [0, 255, 255],  # Classe 5: Cyan
    [128, 128, 128], # Classe 6: Gris
    [255, 255, 255]  # Classe 7: Blanc
]

# 2. Create the app object
app = FastAPI()
best_model = tf.keras.models.load_model("best_model", custom_objects={'dice_coeff': dice_coeff})
second_model = tf.keras.models.load_model("second_model", custom_objects={'dice_coeff': dice_coeff})

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/segmentation/best_model')
async def segment_image_best(image_file: UploadFile = File(...)):
    # Lire l'image téléchargée
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
  
    image = image.resize((256, 256))  # Redimensionner l'image selon les besoins de votre modèle
    image_array = np.array(image)  # Convertir l'image en tableau numpy
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0 
    
    # Effectuer la segmentation d'image à l'aide du modèle
    segmented_image_array = best_model.predict(image_array)
    segmented_image = np.zeros((256,256,3),dtype=np.uint8)
    
    for i in range(segmented_image_array.shape[1]):  # Parcourir les lignes
        for j in range(segmented_image_array.shape[2]):  # Parcourir les colonnes
            idx = np.argmax(segmented_image_array[0, i, j])  # Trouver l'indice de la classe prédite
            segmented_image[i, j] = couleur[idx]  # Assigner la couleur correspondante
    
    # Sauvegarder l'image temporairement
    temp_image_path = "temp_image.png"
    plt.imsave(temp_image_path, segmented_image)

    return FileResponse(temp_image_path, media_type="image/png")
    
@app.post('/segmentation/second_model_old')
async def segment_image_second(image_file: UploadFile = File(...)):
    # Lire l'image téléchargée
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    
  
    image = image.resize((256, 256))  # Redimensionner l'image selon les besoins de votre modèle
    image_array = np.array(image)  # Convertir l'image en tableau numpy
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0 
    
        
    # Effectuer la segmentation d'image à l'aide du modèle
    segmented_image_array = second_model.predict(image_array)
    
    # Taille de l'image d'entrée
    input_height = segmented_image_array.shape[1]
    input_width = segmented_image_array.shape[2]
    
    # Taille de l'image de sortie
    output_height = 256
    output_width = 256
    
    # Créer une image vide de taille 256x256
    segmented_image = np.zeros((output_height,output_width,3),dtype=np.uint8)
    
    # Assigner les couleurs aux pixels en fonction du tableau de prédiction
    scale_height = output_height // input_height
    scale_width = output_width // input_width

    # Assigner les couleurs aux pixels en fonction du tableau de prédiction
    for i in range(input_height):
        for j in range(input_width):
            color_index = int(segmented_image_array[0][i][j] * (len(couleur)-1))
            segmented_image[i*scale_height:(i+1)*scale_height, j*scale_width:(j+1)*scale_width] = couleur[color_index]

    # Sauvegarder l'image temporairement
    temp_image_path = "temp_image.png"
    plt.imsave(temp_image_path, segmented_image)

    return FileResponse(temp_image_path, media_type="image/png")
    #Response(content=segmented_image_bytes.getvalue(), media_type='image/png')
    
@app.post('/segmentation/second_model')
async def segment_image_second(image_file: UploadFile = File(...)):
    # Lire l'image téléchargée
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    
    #Sauvegarde dans un dossier temporaire tmp
        # Enregistrer l'image dans un emplacement fixe
    image_path = "tmp/image_tmp.png"
    image.save(image_path)    
    
    model = vgg_unet(n_classes=8, input_height=256, input_width=512)
    model.load_weights('vgg_unet.h5')
    
    pred = model.predict_segmentation(inp='tmp/image_tmp.png')
    # Convertir l'array numpy en image PIL
    predicted_image = Image.fromarray(pred.astype('uint8'))

    # Redimensionner l'image à la taille désirée (par exemple, 512x256)
    predicted_image_resized = predicted_image.resize((512, 256))
    
    # Sauvegarder l'image temporairement
    temp_image_path = "temp_image.png"
    plt.imsave(temp_image_path, predicted_image_resized)

    return FileResponse(temp_image_path, media_type="image/png")
    #Response(content=segmented_image_bytes.getvalue(), media_type='image/png')

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload