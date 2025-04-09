import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

# Charger les données collectées
data_folder = "sign_data"
X = []
y = []

for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        label = file.split(".")[0].upper()  # Obtenir la lettre
        file_path = os.path.join(data_folder, file)

        # Charger le fichier CSV
        data = pd.read_csv(file_path).values
        X.extend(data)
        y.extend([label] * len(data))  # Associer la lettre à chaque ligne

# Convertir en numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# Vérification des dimensions
print(f"Dimensions de X : {X.shape}")
print(f"Dimensions de y : {y.shape}")

# Normaliser les données (option simple : diviser par la valeur maximale)
X = X / (np.max(X) + 1e-8)

# Encodage des labels (lettres A-Z)
label_encoder = LabelBinarizer()
y = label_encoder.fit_transform(y)

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construire le modèle
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Sauvegarder le modèle
model.save("alphabet_signs_model.h5")
print("Modèle enregistré")

# Évaluer la précision sur les données de test
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"Précision sur les données de test : {accuracy * 100:.2f}%")

# Rapport de classification détaillé
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
