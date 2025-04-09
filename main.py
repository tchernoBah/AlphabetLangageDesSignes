import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import pyttsx3

# Initialisation de MediaPipe et du synthétiseur vocal
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
tts = pyttsx3.init()

# Chargement du modèle
model = tf.keras.models.load_model("alphabet_signs_model.h5")
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialisation webcam
cap = cv2.VideoCapture(0)

last_prediction = None
prediction_start_time = 0
detection_time = 0
pronounced = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).flatten().reshape(1, -1)
            start_time = time.time()  # Début du chronomètre
            prediction = model.predict(landmarks)
            detection_time = time.time() - start_time
            predicted_letter = alphabet[np.argmax(prediction)]


            # Si la lettre est identique à la précédente
            if predicted_letter == last_prediction:
                if time.time() - prediction_start_time >= 1 and not pronounced:
                    tts.say(predicted_letter)
                    tts.runAndWait()
                    pronounced = True
            else:
                last_prediction = predicted_letter
                prediction_start_time = time.time()
                pronounced = False

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Lettre: {predicted_letter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Temps: {detection_time:.2f} sec", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2)
    cv2.putText(frame, "Appuyer sur 'q' pour quitter", (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Reconnaissance et Prononciation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
