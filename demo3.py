import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

students_folder = "Students"
known_face_encodings = []
known_face_names = []  # Student names
known_face_numbers = []  # Student numbers

for student_dir in os.listdir(students_folder):
    dir_path = os.path.join(students_folder, student_dir)
    if os.path.isdir(dir_path):
        parts = student_dir.split('_')
        if len(parts) >= 2:
            student_name = parts[0]
            student_number = parts[1]
        else:
            print(f"Folder name '{student_dir}' does not match expected format 'Name_Number'.")
            continue

        image_found = False
        for file in os.listdir(dir_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dir_path, file)
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(student_name)
                    known_face_numbers.append(student_number)
                    image_found = True
                    break  # Use the first valid image found
        if not image_found:
            print(f"No valid image found in folder '{student_dir}'.")

# -------------------------
# Setup Attendance CSV file
# -------------------------
attendance_filename = f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
if not os.path.exists(attendance_filename):
    with open(attendance_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["No", "Student Name", "Student Number"])

marked_students = set()  # To track which students have been marked

def mark_attendance(student_name, student_number):
    """Add the student to the attendance file if not already marked."""
    if student_number not in marked_students:
        marked_students.add(student_number)
        row_no = len(marked_students)
        with open(attendance_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([row_no, student_name, student_number])

# -------------------------
# Use Video File from Phone
# -------------------------
# Replace "phone_video.mp4" with the path to the video file you transferred from your phone.
video_file = "phone_video.mp4"
cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for both Mediapipe and face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use Mediapipe for face detection
    results = face_detection.process(rgb_frame)
    face_locations = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            # Append face location in (top, right, bottom, left) order
            face_locations.append((y, x + width, y + height, x))
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Compute face encodings for detected faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        student_number = ""
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1

            if best_match_index != -1 and matches[best_match_index]:
                name = known_face_names[best_match_index]
                student_number = known_face_numbers[best_match_index]
                mark_attendance(name, student_number)

        cv2.putText(frame, f"{name} {student_number}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition with Mediapipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
