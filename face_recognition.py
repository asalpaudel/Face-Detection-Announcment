import cv2
import numpy as np
import os
from speech import welcome_person
import threading
import time
import queue

# Setup
cap = cv2.VideoCapture(0)
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml')
profile_cascade_path = os.path.join(script_dir, 'haarcascade_profileface.xml')
dataset_path = os.path.join(script_dir, "face_dataset/")

face_cascade = cv2.CascadeClassifier(cascade_path)
profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

# Create LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Prepare data
faces = []
labels = []
class_id = 0
names = {}
speech_queue = queue.Queue()
recognized_times = {}
spoken_names = set()
detection_counts = {}  # Track consecutive detections

# Dataset preparation
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

# Speech worker thread
def speech_worker():
    while True:
        name = speech_queue.get()
        if name is None:
            break
        welcome_person(f"{name}")
        time.sleep(3)  # Wait before allowing next speech
        speech_queue.task_done()

# Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Start Recognition loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = []

    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_detected.extend(frontal_faces)

    # Detect left profile faces
    left_profiles = profile_cascade.detectMultiScale(gray, 1.3, 5)
    faces_detected.extend(left_profiles)

    # Detect right profile faces (flip image)
    flipped_gray = cv2.flip(gray, 1)
    right_profiles = profile_cascade.detectMultiScale(flipped_gray, 1.3, 5)
    for (x, y, w, h) in right_profiles:
        x = gray.shape[1] - x - w
        faces_detected.append((x, y, w, h))

    current_names_in_frame = set()

    for (x, y, w, h) in faces_detected:
        offset = 10
        x_start = max(x - offset, 0)
        y_start = max(y - offset, 0)
        x_end = min(x + w + offset, gray.shape[1])
        y_end = min(y + h + offset, gray.shape[0])

        face_section = gray[y_start:y_end, x_start:x_end]
        face_section = cv2.resize(face_section, (100, 100))

        label, confidence = recognizer.predict(face_section)

        if confidence < 90:
            name = names[label]
            current_names_in_frame.add(name)
            current_time = time.time()

            detection_counts[name] = detection_counts.get(name, 0) + 1

            if detection_counts[name] >= 3:
                if name not in recognized_times or (current_time - recognized_times[name]) >= 10:
                    recognized_times[name] = current_time
                    if name not in spoken_names:
                        spoken_names.add(name)
                        print(f"[INFO] Queuing speech for: {name} with confidence: {confidence}")
                        speech_queue.put(name)
        else:
            name = "Unknown"

        cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Reset detection counts for names not seen in current frame
    for tracked_name in list(detection_counts.keys()):
        if tracked_name not in current_names_in_frame:
            detection_counts[tracked_name] = 0

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
speech_queue.put(None)
speech_thread.join()
cap.release()
cv2.destroyAllWindows()
