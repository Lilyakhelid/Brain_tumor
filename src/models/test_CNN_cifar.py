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
# %%
# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Précision sur les données de test : {test_accuracy:.2f}')

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
    plt.imshow(explanations[i], cmap="jet", alpha=0.5)  # Superposer l'attribution
    plt.axis('off')
plt.show()