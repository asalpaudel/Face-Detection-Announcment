import cv2
import face_recognition
import pickle
import time
import threading
from gtts import gTTS
import os
import tempfile
from playsound import playsound
from datetime import datetime

# Speak the welcome message in Nepali
def speak_nepali(text):
    try:
        language = 'ne'
        speech = gTTS(text=text, lang=language, slow=False)
        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        speech.save(temp_path)
        playsound(temp_path)
        os.remove(temp_path)
    except Exception as e:
        print(f"❌ Error during TTS: {e}")

# Function to welcome recognized person
def welcome_person(name):
    message = f"Virinchi College ma {name} lai swagat cha."
    threading.Thread(target=speak_nepali, args=(message,)).start()

# Log recognized person with timestamp
def log_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("attendance_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{now}] {name} recognized\n")

# Load known face encodings
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

video_capture = cv2.VideoCapture(0)

# Cooldown tracking
last_spoken = {"name": None, "time": 0}
speak_cooldown = 10  # seconds

# FPS tracking
frame_count = 0
start_time = time.time()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to read from camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = data["names"][matched_idx]

        current_time = time.time()
        if name != "Unknown" and (name != last_spoken["name"] or (current_time - last_spoken["time"]) > speak_cooldown):
            welcome_person(name)
            log_attendance(name)
            last_spoken["name"] = name
            last_spoken["time"] = current_time

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # FPS display
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition + Welcome", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
