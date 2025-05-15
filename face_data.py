import cv2
import numpy as np 
import os
from drive_uploader import authenticate, get_or_create_folder, upload_file

# Connect to HD stream (Channel 101)
cap = cv2.VideoCapture("rtsp://admin:Virinchi%401@192.168.1.10:554/Streaming/Channels/101")
# cap = cv2.VideoCapture(0)  # Use local camera for testing
# Check if stream opened
if not cap.isOpened():
    print("[ERROR] Cannot open RTSP stream")
    exit()

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml')
profile_cascade_path = os.path.join(script_dir, 'haarcascade_profileface.xml')
dataset_path = os.path.join(script_dir, "face_dataset")
os.makedirs(dataset_path, exist_ok=True)

# Load cascades
face_cascade = cv2.CascadeClassifier(cascade_path)
profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

skip = 0
face_data = []

file_name = input("Enter the name of person : ")

def enhance_lighting(gray_img):
    return cv2.equalizeHist(gray_img)

while True:
    for _ in range(5):  # Flush buffer to reduce delay
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = enhance_lighting(gray_frame)

    # Scale ratios to map coordinates back to original HD frame
    scale_x = frame.shape[1] / small_frame.shape[1]
    scale_y = frame.shape[0] / small_frame.shape[0]

    # Detect faces
    faces = []
    faces.extend(face_cascade.detectMultiScale(gray_frame, 1.3, 5))
    faces.extend(profile_cascade.detectMultiScale(gray_frame, 1.3, 5))

    # Detect flipped (right-profile) faces
    flipped_gray = cv2.flip(gray_frame, 1)
    flipped_profiles = profile_cascade.detectMultiScale(flipped_gray, 1.3, 5)
    for (x, y, w, h) in flipped_profiles:
        x = gray_frame.shape[1] - x - w
        faces.append((x, y, w, h))

    if len(faces) == 0:
        cv2.imshow("faces", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        continue

    # Use the largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    skip += 1

    for face in faces[:1]:
        x, y, w, h = face
        # Map back to original frame coordinates
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        offset = 10
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(frame.shape[1], x + w + offset)
        y2 = min(frame.shape[0], y + h + offset)

        face_offset = frame[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face_offset, cv2.COLOR_BGR2GRAY)
        face_gray = enhance_lighting(face_gray)
        face_selection = cv2.resize(face_gray, (100, 100))

        if skip % 2 == 0:
            face_data.append(face_selection)
            print(f"Captured sample: {len(face_data)}")

        cv2.imshow("face", face_selection)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Samples: {len(face_data)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

    if len(face_data) >= 200:
        print("200 samples collected. Exiting...")
        break

# Save dataset
saved_path = os.path.join(dataset_path, file_name + ".npy")
face_data = np.array(face_data).reshape((len(face_data), -1))
np.save(saved_path, face_data)
print(f"Dataset saved at: {saved_path}")

# Upload to Google Drive
print("[Drive] Uploading to Google Drive...")
service = authenticate()
folder_id = get_or_create_folder(service, 'face_dataset')
upload_file(service, saved_path, folder_id)

cap.release()
cv2.destroyAllWindows()
