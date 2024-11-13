
# Ici, récupérer le prétraitement qui a été fait, puis mettre en place le modèle simplement et l'enregistrer.
# Essayer de bien nommer le modèle avec les hyperparamètres et faire ce script le plus réutilisable possible pour pouvoir l'utiliser plusieurs fois et créer un benchmark.

# %%
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from module_for_preprocessing import *  # fonctions de prétraitement
from tensorflow.keras import layers, models
import yaml
from datetime import datetime  # Nécessaire pour générer un timestamp unique


with open('../../config.yml', 'r') as file:
    config = yaml.safe_load(file)
size = config['données']['image']['size']

# %%


training = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Training")
validation = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Validation")
testing = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data", "Testing")

x_train, y_train = load_images_with_preprocessing(training, size)
x_val, y_val = load_images_with_preprocessing(validation, size)
x_test, y_test = load_images_with_preprocessing(testing, size)

# %%

model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),

    # Bloc 1
    layers.Conv2D(32, (3, 3), activation='relu'),

    # Bloc 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Bloc 3
    layers.Conv2D(128, (3, 3), activation='relu'),

    # Bloc 4
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Couche Fully Connected
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),  # Pour éviter le surapprentissage
    layers.Dense(4, activation='softmax')

])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()


history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)

# Évaluation du modèle sur le jeu de test
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


base_dir = os.getcwd()  
save_dir = os.path.join(base_dir, 'sauvegardes_modeles')  
os.makedirs(save_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas

# Générer un nom de fichier unique avec la date et l'heure pour reconnaitre les modeles
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(save_dir, f"modele_brain_tumor_{timestamp}.h5")
model.save(model_path)
print(f"Modèle sauvegardé dans : {model_path}")
