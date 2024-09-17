import sys
sys.path.append('E:/project')
import pygame

import cv2
import numpy as np
import tensorflow as tf
import sys
import requests
from datetime import datetime


class MaskDetector:
    def __init__(self, model_path, alert_sound_path):
        self.model = tf.keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.alert_sound = cv2.imread(alert_sound_path, cv2.IMREAD_UNCHANGED)

    def detect_mask(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                cv2.putText(frame, "Looks good", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                return frame

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                processed_face = preprocess(face)
                prediction = self.model.predict(processed_face)
                label = "Scarf alert" if prediction[0][0] > 0.5 else "Looks good"

                if label == "Scarf alert":
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    self.raise_alert()
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return frame

    def raise_alert(self):
        now = datetime.now()
        alert_text = f"Scarf alert generated at {now}"
        # Append alert text to a file
        with open("alerts.txt", "a") as file:
            file.write(alert_text + "\n")
        bot_token = '6149367572:AAHK6f19zie10OXbEylyQbNXwA3E2xeWYsE'
        chat_id = '713496494'  # The chat ID of the user or group you want to send the message to
        message_text = f'A Person with Scarf detected under the camera. \n Time: {now}'

        # Call the function to send the message
        self.send_message(chat_id, message_text, bot_token)

    def send_message(self, chat_id, text, token):
        url = f"https://api.telegram.org/bot6149367572:AAHK6f19zie10OXbEylyQbNXwA3E2xeWYsE/sendMessage"
        params = {
            "chat_id": chat_id,
            "text": text
        }
        response = requests.post(url, params=params)
        return response.json()

def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image