import sys
sys.path.append('E:/project/')
import cv2
import requests
from flask import Flask, render_template, Response
from models.yolov3 import YOLOv3
from models.detect_mask import MaskDetector
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


app = Flask(__name__)

yolo_model = YOLOv3(weights_path='models/yolov3.weights', config_path='models/yolov3.cfg', labels_path='models/coco.names')
mask_model = MaskDetector('models/mask_detector.model', 'alert_sound.wav')

weapon_camera = None
mask_camera = None
detection_type = None

def gen_frames(camera):
    while True:
        if detection_type is None:
            break

        success, frame = camera.read()
        if not success:
            break
        else:
            alert = False
            if detection_type == 'weapons':
                boxes, confidences, class_ids = yolo_model.detect_objects(frame)
                alert = yolo_model.detect_alert(boxes, class_ids)
                frame = yolo_model.draw_boxes(frame, boxes, confidences, class_ids, alert=alert)
            elif detection_type == 'masks':
                frame = mask_model.detect_mask(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            if alert:
                yolo_model.play_alert_sound()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n' + b'alert')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Watchdog event handler
class FileModifiedEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path == file_path:
            with open(file_path, 'r') as file:
                new_line = file.readlines()[-1].strip()  # Get the last line added
                # send_message(new_line)  # Send message to the Telegram bot

                # Move the file pointer to the end
                file.seek(0, 2)

# # Send message to Telegram bot
# def send_message(text):
#     url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
#     params = {
#         "chat_id": telegram_chat_id,
#         "text": text
#     }
#     response = requests.post(url, params=params)
#     return response.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_weapons')
def video_feed_weapons():
    global weapon_camera, detection_type
    detection_type = 'weapons'
    weapon_camera = cv2.VideoCapture(0)
    return Response(gen_frames(weapon_camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_masks')
def video_feed_masks():
    global mask_camera, detection_type
    detection_type = 'masks'
    mask_camera = cv2.VideoCapture(0)
    return Response(gen_frames(mask_camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    global weapon_camera, mask_camera, detection_type
    detection_type = None

    if weapon_camera is not None:
        weapon_camera.release()
    if mask_camera is not None:
        mask_camera.release()

    return 'Stopped detection'

if __name__ == '__main__':
    # Create watchdog observer
    observer = Observer()
    event_handler = FileModifiedEventHandler()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
