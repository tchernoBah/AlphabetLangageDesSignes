import cv2
import mediapipe as mp
import csv
import os

# Configuration de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dossier pour stocker les donnéesc
data_folder = "sign_data"
os.makedirs(data_folder, exist_ok=True)

# Entrée de l'utilisateur pour choisir la lettre
letter = input("Lettre actuelle (A-Z) : ").upper()

# Préparer le fichier CSV
csv_file = open(f"{data_folder}/{letter}.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([f"point_{i}_{dim}" for i in range(21) for dim in ['x', 'y', 'z']])  # En-têtes des colonnes

# Ouvrir la webcam
cap = cv2.VideoCapture(0)
print(f"Commencez à signer la lettre '{letter}' (appuyez sur 'q' pour quitter).")

while True:
    success, frame = cap.read()
    if not success:
        print("Erreur de capture vidéo !")
        break

    # Convertir en RGB (MediaPipe nécessite un format RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détecter les mains
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les points clés sur l'image
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraire les coordonnées des points clés
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Ajouter x, y, z
            csv_writer.writerow(landmarks)

    # Afficher la vidéo avec les points détectés
    cv2.imshow("Collecte de données", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()