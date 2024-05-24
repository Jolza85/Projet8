# -*- coding: utf-8 -*-
# 1. Library imports
import uvicorn
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

# 2. Create the app object
app = FastAPI()
load_model = tf.keras.models.load_model("best_model", custom_objects={'dice_coeff': dice_coeff})

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/segmentation')
async def segment_image(image_file: UploadFile = File(...)):
    # Lire l'image téléchargée
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    
  
    image = image.resize((128, 128))  # Redimensionner l'image selon les besoins de votre modèle
    image_array = np.array(image)  # Convertir l'image en tableau numpy
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0 
    
    # Effectuer la segmentation d'image à l'aide du modèle
    segmented_image_array = load_model.predict(image_array)[0]

    #colored_predictions = label_to_color_image(np.argmax(segmented_image_array, axis=-1)) 
    # Convertir l'array numpy en image PIL
    #segmented_image = Image.fromarray((segmented_image_array * 255).astype(np.uint8), mode='RGB')
    
    plt.imshow(np.argmax(segmented_image_array, axis=-1))
    plt.title('Prédictions de segmentation')
    plt.axis('off')
    
    # Sauvegarder l'image temporairement
    temp_image_path = "temp_image.png"
    plt.savefig(temp_image_path)
    plt.close()

    # Convertir l'image segmentée en format accepté par FastAPI
    #segmented_image_bytes = io.BytesIO()
    #segmented_image.save(segmented_image_bytes, format="PNG")
    #segmented_image_bytes.seek(0)

    return FileResponse(temp_image_path, media_type="image/png")
    #Response(content=segmented_image_bytes.getvalue(), media_type='image/png')

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload