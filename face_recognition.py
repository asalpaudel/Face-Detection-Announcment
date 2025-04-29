import cv2
import numpy as np
import os

# Setup
cap = cv2.VideoCapture(0)
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml')
dataset_path = os.path.join(script_dir, "face_dataset/")
face_cascade = cv2.CascadeClassifier(cascade_path)

# Create LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare data
faces = []
labels = []
class_id = 0
names = {}

# Dataset preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(os.path.join(dataset_path, fx))

        # Check if color or grayscale
        if data_item.shape[1] == 30000:
            # Color image, reshape accordingly
            data_item = data_item.reshape((-1, 100, 100, 3))
            # Convert color to grayscale
            data_item = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in data_item])
        elif data_item.shape[1] == 10000:
            # Already grayscale
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

# Train LBPH recognizer
recognizer.train(faces, labels)

print("[INFO] Training complete!")

# Start Recognition
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        offset = 10
        x_start = max(x - offset, 0)
        y_start = max(y - offset, 0)
        x_end = min(x + w + offset, gray.shape[1])
        y_end = min(y + h + offset, gray.shape[0])

        face_section = gray[y_start:y_end, x_start:x_end]
        face_section = cv2.resize(face_section, (100, 100))

        label, confidence = recognizer.predict(face_section)

        # Lower confidence -> better match
        if confidence < 80:  # You can tune this threshold
            name = names[label]
        else:
            name = "Unknown"

        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
