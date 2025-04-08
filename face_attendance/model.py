import cv2
import face_recognition
import pickle
from datetime import datetime
import os

# Load known face encodings
with open("face_attendance/encodings/faces.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Initialize video capture (0 = webcam; or use a video file)
cap = cv2.VideoCapture(0)

# Attendance file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Time\n")

recorded_names = set()  # to avoid multiple entries

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

            # Mark attendance only once per session
            if name not in recorded_names:
                recorded_names.add(name)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, "a") as f:
                    f.write(f"{name},{now}\n")

        # Draw box & name
        top, right, bottom, left = [v * 4 for v in location]  # scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
