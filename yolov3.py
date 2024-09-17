import cv2
import sys
import numpy as np
import pygame
from datetime import datetime
import requests
import pyautogui
import io
sys.path.append('E:/project')

class YOLOv3:
    def __init__(self, weights_path, config_path, labels_path, confidence_threshold=0.5, nms_threshold=0.3):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.labels = open(labels_path).read().strip().split("\n")
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        pygame.mixer.init()  # Initialize Pygame mixer
        self.alert_sound = pygame.mixer.Sound('alert_sound.wav')

    def draw_boxes(self, frame, boxes, confidences, class_ids, alert=False):
        for i, box in enumerate(boxes):
            x, y, w, h = box
            label = self.labels[class_ids[i]]
            confidence = confidences[i]

            if alert:
                color = (0, 0, 255)  # Red color for alert
                self.alert_sound.play()  # Play the alert sound
            else:
                color = (0, 255, 0)  # Green color for normal detection

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        indices = np.array(indices).flatten()
        filtered_boxes = [boxes[i] for i in indices]
        filtered_confidences = [confidences[i] for i in indices]
        filtered_class_ids = [class_ids[i] for i in indices]

        return filtered_boxes, filtered_confidences, filtered_class_ids

    def detect_alert(self, boxes, class_ids):
        for i, class_id in enumerate(class_ids):
            label = self.labels[class_id]
            if label in ['weapon', 'knife', 'gun']:
                self.save_alert(label)
                return True
        return False

    def send_message(self, chat_id, text, token):
        url = f"https://api.telegram.org/bot6149367572:AAHK6f19zie10OXbEylyQbNXwA3E2xeWYsE/sendMessage"
        params = {
            "chat_id": chat_id,
            "text": text
        }
        response = requests.post(url, params=params)
        return response.json()

    def save_alert(self, label):
        now = datetime.now()
        alert_text = f"{label} alert generated at {now}"
        # Append alert text to a file
        with open("alerts.txt", "a") as file:
            file.write(alert_text + "\n")
        bot_token = '6149367572:AAHK6f19zie10OXbEylyQbNXwA3E2xeWYsE'
        chat_id = '713496494'  # The chat ID of the user or group you want to send the message to
        message_text = f'A person with Knife detected under the Camera.\n Time: {now}'

        # Call the function to send the message
        self.send_message(chat_id, message_text, bot_token)

#send telegram alert with image if we have telegram premium

    # def send_message(self, chat_id, text, token, image=None):
    #     url = f"https://api.telegram.org/bot6104944122:AAE_d4zsCJhE7EUaDnS-MRLtCHpzO_zrHN4/sendMessage"
    #     params = {
    #         "chat_id": chat_id,
    #         "text": text
    #     }

    #     files1 = {}
    #     if image is not None:
    #         files['photo'] = image

    #     response = requests.post(url, params=params, files=files1)
    #     return response.json()

    # def save_alert(self, label):
    #     now = datetime.now()
    #     alert_text = f"{label} alert generated at {now}"
    #     # Append alert text to a file
    #     with open("alerts.txt", "a") as file:
    #         file.write(alert_text + "\n")
    #     bot_token = '6104944122:AAE_d4zsCJhE7EUaDnS-MRLtCHpzO_zrHN4'
    #     chat_id = '6276902856'  # The chat ID of the user or group you want to send the message to
    #     message_text = f"Weapon detected at {now}"

    #     # Take a screenshot
    #     screenshot = pyautogui.screenshot()

    #     # Create an in-memory byte stream
    #     image_stream = io.BytesIO()
    #     screenshot.save(image_stream, format='PNG')
    #     image_stream.seek(0)

    #     # Send the screenshot image with the message
    #     self.send_message(chat_id, message_text, bot_token, image_stream)

    def play_alert_sound(self):
        self.alert_sound.play()

