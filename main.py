import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialisation de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Charger un modèle pré-entraîné (un fichier .h5 de reconnaissance d'alphabet)
model = tf.keras.models.load_model("alphabet_signs_model.h5")

# Liste des lettres de l'alphabet
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convertir en RGB (MediaPipe nécessite un format RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détecter les mains
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraire les coordonnées des points clés
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Convertir en un format compatible pour le modèle
            landmarks = np.array(landmarks).flatten().reshape(1, -1)

            # Prédire la lettre
            prediction = model.predict(landmarks)
            predicted_letter = alphabet[np.argmax(prediction)]

            # Dessiner les points clés sur l'image
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Afficher la lettre prédite
            cv2.putText(frame, f"Lettre: {predicted_letter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Afficher la vidéo
    cv2.imshow("Reconnaissance des signes", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()