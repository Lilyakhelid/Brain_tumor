import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as app
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import streamlit as st
from PIL import Image

# Définir les noms des classes
class_names = {
    0: "Gliome",
    1: "Méningiome",
    2: "Pas de tumeur",
    3: "Pituitary",
}


# Charger le modèle sauvegardé
@st.cache_resource
def load_cnn_model():
    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, "sauvegardes_modeles")
    model_to_load = "modele_brain_tumor_20241117_205951.h5"
    model_path = os.path.abspath(os.path.join(save_dir, model_to_load))

    model = load_model(model_path)
    st.write(f"Modèle chargé : {model_path}")
    st.write(f"Dimensions d'entrée attendues : {model.input_shape}")
    return model


model = load_cnn_model()


# Fonction pour prédire la classe d'une image téléchargée
def predict_image_class(image, model, target_size=(256, 256)):
    # Redimensionner et prétraiter l'image
    img = image.resize(target_size)
    img_array = img_to_array(img)  # Convertir l'image en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_array /= 255.0  # Normaliser les pixels entre 0 et 1

    # Prédire la classe
    predictions = model.predict(img_array)
    print(predictions)
    predicted_class = np.argmax(
        predictions, axis=1
    )  # Trouver l'indice de la classe avec la probabilité la plus élevée

    return class_names.get(predicted_class[0])  # Retourner la classe prédite


# Interface Streamlit
st.title("Classification d'image avec un modèle CNN")
st.write("Téléchargez une image pour prédire sa classe.")

# Option de téléchargement de fichier
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image téléchargée
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Image téléchargée", use_column_width=True)
    st.write("Classification en cours...")

    # Effectuer la prédiction
    classe_predite = predict_image_class(image, model)
    st.write(f"La classe prédite pour l'image est : {classe_predite}")

# streamlit run CNN_brain_tumor_saved.py
# image(15).jpg testing meningiome
