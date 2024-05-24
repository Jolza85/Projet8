from flask import Flask, render_template, request, send_file, send_from_directory
import requests
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

app = Flask(__name__, static_folder='static')

# Définir l'URL de votre API FastAPI
API_URL = 'https://projet8.onrender.com'

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin vers le dossier contenant les images et les masques
datas_directory = os.path.join(os.getcwd(), 'datas')

# Charger la palette nipy_spectral
palette = np.array(plt.cm.get_cmap('nipy_spectral')(np.arange(256))) * 255
palette = palette.astype(np.uint8)

@app.route('/')
def index():
    # Récupérer la liste des fichiers dans le dossier static/images
    image_folder = os.path.join(app.static_folder, 'images')
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    return render_template('index.html', image_files=image_files)

@app.route('/segment_image', methods=['POST'])
def segment_image():
    # Récupérer le nom de l'image sélectionnée depuis la requête
    selected_image = request.form['image']
    
    # Construire le chemin complet de l'image
    image_path = f'static/images/{selected_image}'
    
    #logger.info('Image file', image_path)
    # Ouvrir et lire le fichier image
    with open(image_path, 'rb') as file:
        image_data = file.read()
    
    # Envoyer le fichier image à votre API FastAPI
    files = {'image_file': image_data}
    
    response = requests.post(f'{API_URL}/segmentation/best_model', files=files)

    logger.info('code status %s', response.status_code)
    
    # Vérifier le statut de la réponse
    if response.status_code == 200:
        # Afficher l'image segmentée
        segmented_image = BytesIO(response.content)
        return send_file(segmented_image, mimetype='image/png')
    else:
        return 'Error processing image', 500
    
# Endpoint pour servir les masques
@app.route('/datas/masks/test_clean/<path:filename>')
def get_mask(filename):
    
    # Chemin complet vers l'image en niveaux de gris
    grayscale_image_path = os.path.join('static', 'masks', filename)
    
    # Charger l'image en niveaux de gris
    grayscale_image = Image.open(grayscale_image_path)
    
    # Convertir l'image en numpy array
    grayscale_array = np.array(grayscale_image)
    
    # Appliquer la palette nipy_spectral à l'image
    colored_array = palette[grayscale_array]
    
    # Convertir le tableau numpy en image PIL
    colored_image = Image.fromarray(colored_array.astype(np.uint8))
    
    # Convertir l'image en bytes
    output_buffer = BytesIO()
    colored_image.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    
    return send_file(output_buffer, mimetype='image/png')
    #return send_from_directory(os.path.join(datas_directory, 'masks', 'train_clean_normalized'), filename)
    
if __name__ == '__main__':
    app.run(debug=True)