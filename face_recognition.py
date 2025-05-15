import cv2
import numpy as np
import os
from speech import welcome_person
import threading
import time
import queue

# RTSP camera stream from main channel (HD)
cap = cv2.VideoCapture("rtsp://admin:Virinchi%401@192.168.1.10:554/Streaming/Channels/101")
if not cap.isOpened():
    print("[ERROR] Failed to open RTSP stream.")
    exit()

# Paths and classifiers
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml')
profile_cascade_path = os.path.join(script_dir, 'haarcascade_profileface.xml')
dataset_path = os.path.join(script_dir, "face_dataset/")

face_cascade = cv2.CascadeClassifier(cascade_path)
profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

# Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Load dataset
faces = []
labels = []
class_id = 0
names = {}
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(os.path.join(dataset_path, fx))

        if data_item.shape[1] == 30000:
            data_item = data_item.reshape((-1, 100, 100, 3))
            data_item = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data_item])
        elif data_item.shape[1] == 10000:
            data_item = data_item.reshape((-1, 100, 100))
        else:
            print(f"[WARNING] Unexpected shape for {fx}: {data_item.shape}")
            continue

        for face in data_item:
            faces.append(face)
            labels.append(class_id)

        class_id += 1

faces = np.array(faces, dtype=np.uint8)
labels = np.array(labels)

recognizer.train(faces, labels)
print("[INFO] Training complete!")

# Speech thread
speech_queue = queue.Queue()
recognized_times = {}
detection_counts = {}
def speech_worker():
    while True:
        name = speech_queue.get()
        if name is None:
            break
        welcome_person(name)
        time.sleep(3)
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Main loop
while True:
    for _ in range(5):  # Flush RTSP buffer
        cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        continue

    # Resize for detection only
    small_frame = cv2.resize(frame, (640, 480))
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.equalizeHist(gray_small)

    # Scale for mapping coords back to HD
    scale_x = frame.shape[1] / small_frame.shape[1]
    scale_y = frame.shape[0] / small_frame.shape[0]

    # Face detection
    faces_detected = []
    faces_detected.extend(face_cascade.detectMultiScale(gray_small, 1.3, 5))
    faces_detected.extend(profile_cascade.detectMultiScale(gray_small, 1.3, 5))

    flipped_gray = cv2.flip(gray_small, 1)
    flipped_profiles = profile_cascade.detectMultiScale(flipped_gray, 1.3, 5)
    for (x, y, w, h) in flipped_profiles:
        x = gray_small.shape[1] - x - w
        faces_detected.append((x, y, w, h))

    current_names_in_frame = set()

    for (x, y, w, h) in faces_detected:
        # Map to HD
        x_hd = int(x * scale_x)
        y_hd = int(y * scale_y)
        w_hd = int(w * scale_x)
        h_hd = int(h * scale_y)

        offset = 10
        x1 = max(0, x_hd - offset)
        y1 = max(0, y_hd - offset)
        x2 = min(x_hd + w_hd + offset, frame.shape[1])
        y2 = min(y_hd + h_hd + offset, frame.shape[0])

        face_region = frame[y1:y2, x1:x2]
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.equalizeHist(gray_face)
        face_resized = cv2.resize(gray_face, (100, 100))

        label, confidence = recognizer.predict(face_resized)
        name = "Unknown"

        if confidence < 90:
            name = names[label]
            current_names_in_frame.add(name)
            current_time = time.time()
            detection_counts[name] = detection_counts.get(name, 0) + 1

            if detection_counts[name] >= 5 and name not in recognized_times:
                recognized_times[name] = current_time
                print(f"[INFO] Greeting queued: {name} (Confidence: {confidence:.2f})")
                speech_queue.put(name)

        else:
            confidence = 100  # Display high number for unknowns

        cv2.putText(frame, f"{name} ({int(confidence)})", (x_hd, y_hd - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (x_hd, y_hd), (x_hd + w_hd, y_hd + h_hd), (255, 255, 255), 2)

    # Decay detection counts if not seen
    for name in list(detection_counts.keys()):
        if name not in current_names_in_frame:
            detection_counts[name] = max(0, detection_counts[name] - 1)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
speech_queue.put(None)
speech_thread.join()
cap.release()
cv2.destroyAllWindows()
