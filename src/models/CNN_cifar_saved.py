#Ce module sert a découvrir le CNN avec une application utilisant les données cifar (classique)
#ici on apprend a load un model sauvegardé
# %%
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from xplique.attributions import GradCAM
import tensorflow.keras.applications as app
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import streamlit as st
from PIL import Image


class_names = {
    0: "Avion",
    1: "Automobile",
    2: "Oiseau",
    3: "Chat",
    4: "Cerf",
    5: "Chien",
    6: "Grenouille",
    7: "Cheval",
    8: "Bateau",
    9: "Camion"
}

# Définir le chemin vers le modèle sauvegardé
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, 'sauvegardes_modeles')
model_to_load = 'modele_cifar10_20241101_163923.h5' # Remplacez par le nom de votre modèle
model_path = os.path.join(save_dir, model_to_load) 

# Charger le modèle sauvegardé
model = load_model(model_path)

# Fonction pour prédire la classe d'une image téléchargée
def predict_image_class(image, model, target_size=(32, 32)):
    # Redimensionner et prétraiter l'image
    img = image.resize(target_size)
    img_array = img_to_array(img)  # Convertir l'image en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_array /= 255.0  # Normaliser les pixels entre 0 et 1

    # Prédire la classe
    predictions = model.predict(img_array)
    print(predictions)
    predicted_class = np.argmax(predictions, axis=1)  # Trouver l'indice de la classe avec la probabilité la plus élevée

    return class_names.get(predicted_class[0])  # Retourner la classe prédite

# Interface Streamlit
st.title("Classification d'image avec un modèle CNN")
st.write("Téléchargez une image pour prédire sa classe.")

# Option de téléchargement de fichier
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    st.write("Classification en cours...")

    # Effectuer la prédiction
    classe_predite = predict_image_class(image, model)
    st.write(f"La classe prédite pour l'image est : {classe_predite}")

#streamlit run CNN_cifar_saved.py