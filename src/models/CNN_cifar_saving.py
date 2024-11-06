#Ce module sert a découvrir le CNN avec une application utilisant les données cifar (classique)
# %%
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from xplique.attributions import GradCAM
import tensorflow.keras.applications as app
import matplotlib.pyplot as plt
from datetime import datetime
import os
# %%
# Charger et préparer les données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normaliser les images (valeurs entre 0 et 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Encoder les labels en catégories (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# %%
# Définir l'architecture du modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes pour CIFAR-10
])
# %%
# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# %%
# Afficher l'architecture du modèle
model.summary()
# %%
# Entraînement du modèle
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)


# Définir le répertoire de base pour la sauvegarde
base_dir = os.getcwd()  # Utilise le répertoire actuel
save_dir = os.path.join(base_dir, 'sauvegardes_modeles')  # Sous-dossier pour les sauvegardes
os.makedirs(save_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas

# Générer un nom de fichier unique avec la date et l'heure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(save_dir, f"modele_cifar10_{timestamp}.h5")


# Sauvegarder le modèle
model.save(model_path)
print(f"Modèle sauvegardé dans : {model_path}")
#hello