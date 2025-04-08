import cv2
import face_recognition
import pickle
import os


encodings_path = "encodings/faces.pkl"

# Initialize
known_encodings = []
known_names = []

# Load video
video_capture = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process every 5th frame for efficiency
    if frame_count % 5 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            known_encodings.append(face_encoding)
            known_names.append("Sailappan")  # Label it with your name

    frame_count += 1

video_capture.release()

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
os.makedirs(os.path.dirname(encodings_path), exist_ok=True)

with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encoding completed and saved to {encodings_path}")
