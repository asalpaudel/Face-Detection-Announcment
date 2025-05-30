import cv2
import numpy as np
import os
from speech import welcome_person
import threading
import time
import queue
from drive_uploader import authenticate, get_or_create_folder, list_files_in_folder
from googleapiclient.http import MediaIoBaseDownload
import io


def sync_and_retrain():
    service = authenticate()
    folder_id = get_or_create_folder(service, 'face_dataset')
    known_files = {}

    global recognizer, names

    while True:
        try:
            files = list_files_in_folder(service, folder_id)
            updated = False

            for f in files:
                fname = f['name']
                fid = f['id']
                modified = f['modifiedTime']

                if fname.endswith('.npy'):
                    local_path = os.path.join(dataset_path, fname)

                    # Check if file is new or modified
                    if fname not in known_files or known_files[fname] != modified:
                        print(f"[Sync] Downloading: {fname}")
                        request = service.files().get_media(fileId=fid)
                        fh = io.FileIO(local_path, 'wb')
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()

                        known_files[fname] = modified
                        updated = True

            if updated:
                print("[Sync] Changes detected. Re-training recognizer.")
                # Reload & retrain
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
                            continue

                        for face in data_item:
                            faces.append(face)
                            labels.append(class_id)

                        class_id += 1

                if faces:
                    recognizer.train(np.array(faces, dtype=np.uint8), np.array(labels))
                    print("[INFO] Initial training complete.")
                else:
                    print("[INFO] No local data yet. Waiting for sync...")

        except Exception as e:
            print(f"[Sync Error] {e}")

        time.sleep(30)  # check every 30 seconds
        

cap = cv2.VideoCapture("rtsp://admin:Virinchi%401@192.168.1.10:554/Streaming/Channels/101")
#cap = cv2.VideoCapture("rtsp://admin:Admin%40123@192.168.1.12:554/Streaming/Channels/102")

# cap = cv2.VideoCapture(0)  # Use local camera for testing

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

if len(faces) > 0:
    recognizer.train(faces, labels)
    print("[INFO] Training complete!")
else:
    print("[INFO] No local data yet. Waiting for sync to populate face_dataset...")
    trained = False


sync_thread = threading.Thread(target=sync_and_retrain, daemon=True)
sync_thread.start()

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

    orig_h, orig_w = frame.shape[:2]

    # Resize for detection only
    small_frame = cv2.resize(frame, (640, 480))
    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.equalizeHist(gray_small)

    scale_x = orig_w / 640
    scale_y = orig_h / 480

    # Detect faces in resized (small) frame
    faces_detected = []
    faces_detected.extend(face_cascade.detectMultiScale(gray_small, 1.3, 5))
    faces_detected.extend(profile_cascade.detectMultiScale(gray_small, 1.3, 5))

    # Detect flipped profile faces
    flipped_gray = cv2.flip(gray_small, 1)
    flipped_profiles = profile_cascade.detectMultiScale(flipped_gray, 1.3, 5)
    for (x, y, w, h) in flipped_profiles:
        x = gray_small.shape[1] - x - w
        faces_detected.append((x, y, w, h))

    current_names_in_frame = set()

    for (x, y, w, h) in faces_detected:
        # Map small-frame coords to original frame
        x_hd = int(x * scale_x)
        y_hd = int(y * scale_y)
        w_hd = int(w * scale_x)
        h_hd = int(h * scale_y)

        offset = 10
        x1 = max(0, x_hd - offset)
        y1 = max(0, y_hd - offset)
        x2 = min(x_hd + w_hd + offset, orig_w)
        y2 = min(y_hd + h_hd + offset, orig_h)

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

            if detection_counts[name] >= 10 and name not in recognized_times:
                recognized_times[name] = current_time
                print(f"[INFO] Greeting queued: {name} (Confidence: {confidence:.2f})")
                speech_queue.put(name)
        else:
            confidence = 100  # Show 100 for unknowns

        # Draw corrected bounding box and label
        cv2.putText(frame, f"{name} ({int(confidence)})", (x_hd, y_hd - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x_hd, y_hd), (x_hd + w_hd, y_hd + h_hd), (255, 255, 255), 2)

    # Decay detection counts
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
