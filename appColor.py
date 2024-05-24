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
from keras_segmentation.models.fcn import fcn_8_vgg

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

# 2. Create the app object
app = FastAPI()
best_model = tf.keras.models.load_model("best_model", custom_objects={'dice_coeff': dice_coeff})

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
    segmented_image = np.argmax(segmented_image_array, axis=-1)
    segmented_image = np.expand_dims(segmented_image, axis=-1)
    segmented_image = np.squeeze(segmented_image)
        
    # Sauvegarder l'image temporairement
    temp_image_path = "image_segmented_BM.png"
    plt.imsave(temp_image_path, segmented_image)

    return FileResponse(temp_image_path, media_type="image/png")
    
@app.post('/segmentation/second_model')
async def segment_image_second(image_file: UploadFile = File(...)):
    # Lire l'image téléchargée
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    
    #Sauvegarde dans un dossier temporaire tmp
        # Enregistrer l'image dans un emplacement fixe
    image_path = "tmp/image_tmp.png"
    image.save(image_path)    
    
    model = fcn_8_vgg(n_classes=8 , input_height=256, input_width=512)
    model.load_weights('vgg_fcn.h5')
    
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