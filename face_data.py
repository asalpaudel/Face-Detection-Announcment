import cv2
import numpy as np 
import os

cap = cv2.VideoCapture(0)

script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_alt.xml')
profile_cascade_path = os.path.join(script_dir, 'haarcascade_profileface.xml')
dataset_path = os.path.join(script_dir, "face_dataset")

face_cascade = cv2.CascadeClassifier(cascade_path)
profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

skip = 0
face_data = []

file_name = input("Enter the name of person : ")

def enhance_lighting(gray_img):
    """ Apply histogram equalization to improve lighting in grayscale image """
    return cv2.equalizeHist(gray_img)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = enhance_lighting(gray_frame)

    faces = []

    # Detect frontal faces
    frontal_faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces.extend(frontal_faces)

    # Detect left profile faces
    profile_faces = profile_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces.extend(profile_faces)

    # Detect right profile faces (flipped image)
    flipped_gray = cv2.flip(gray_frame, 1)
    flipped_profiles = profile_cascade.detectMultiScale(flipped_gray, 1.3, 5)
    for (x, y, w, h) in flipped_profiles:
        x = gray_frame.shape[1] - x - w  # flip x back
        faces.append((x, y, w, h))

    if len(faces) == 0:
        cv2.imshow("faces", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        continue

    # Process the largest detected face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    skip += 1

    for face in faces[:1]:
        x, y, w, h = face
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
face_data = np.array(face_data).reshape((len(face_data), -1))
np.save(os.path.join(dataset_path, file_name), face_data)
print(f"Dataset saved at: {dataset_path + '/' + file_name}.npy")

cap.release()
cv2.destroyAllWindows()
