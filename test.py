import unittest
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import time


# Charger le modèle
class TestRecognitionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tf.keras.models.load_model("alphabet_signs_model.h5")
        cls.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        cls.mp_hands = mp.solutions.hands
        cls.hands = cls.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    def test_model_loading(self):
        """ Vérifie que le modèle est bien chargé. """
        self.assertIsNotNone(self.model, "Échec du chargement du modèle")


    def test_hand_detection(self):
        """ Vérifie la détection des mains avec la webcam. """
        cap = cv2.VideoCapture(0)
        self.assertTrue(cap.isOpened(), "Échec d'ouverture de la webcam")

        success, frame = cap.read()
        self.assertTrue(success, "Capture vidéo échouée")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        cap.release()
        cv2.destroyAllWindows()

        self.assertIsNotNone(results, "Échec de la détection des mains")

    def test_detection_time(self):
        """ Vérifie que le temps de détection reste sous une limite acceptable. """
        sample_input = np.random.rand(1, 63)
        start_time = time.time()
        _ = self.model.predict(sample_input)
        detection_time = time.time() - start_time
        self.assertLess(detection_time, 0.5, "Temps de détection trop long")


if __name__ == "__main__":
    unittest.main()
