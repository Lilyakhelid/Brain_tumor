
#Ce module sert a découvrir le CNN avec une application utilisant les données cifar (classique)
#ici on apprend a load un model sauvegardé
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
from tensorflow.keras.models import load_model
import os












# %%
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, 'sauvegardes_modeles')
print(os.listdir(save_dir))
model_to_load = 'modele_cifar10_20241101_163923.h5' #os.listdir(save_dir)[0]
print(f"Chargement du modèle : {model_to_load}")
# %%
model_path = os.path.join(save_dir, model_to_load) 
model = load_model(model_path)

# %%

model.summary()

# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Précision sur les données de test : {test_accuracy:.2f}')

print(f'Précision sur les données de test : {test_loss:.2f}')







# %%

# Charger et préparer les données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normaliser les images (valeurs entre 0 et 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Encoder les labels en catégories (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# %%
# Utiliser GradCAM pour visualiser les zones importantes pour les prédictions du modèle
# Charger un sous-ensemble de données d'image pour tester l'explication GradCAM
X = x_test[:5]  # Par exemple, prends les 5 premières images de test
Y = y_test[:5]

# Créer un explainer GradCAM pour ton modèle
explainer = GradCAM(model) # boucler sur tout les diff explainer.

# Générer les explications pour les images sélectionnées
explanations = explainer.explain(X, Y)

# Afficher les explications superposées sur les images d'origine
for i in range(len(X)):
    plt.subplot(1, len(X), i + 1)
    plt.imshow(X[i])  # Afficher l'image d'origine
 #   plt.imshow(explanations[i], cmap="jet", alpha=0.5)  # Superposer l'attribution
    plt.axis('off')
plt.show()
for i in range(len(X)):
    plt.subplot(1, len(X), i + 1)
    plt.imshow(X[i])  # Afficher l'image d'origine
    plt.imshow(explanations[i], cmap="jet", alpha=0.5)  # Superposer l'attribution
    plt.axis('off')
plt.show()



# %%
