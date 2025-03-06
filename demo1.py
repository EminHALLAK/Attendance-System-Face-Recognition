import cv2
import mediapipe as mp
import face_recognition
import numpy as np

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

known_face_encodings = []
known_face_names = ["emin", "wisam"]
known_images = ["emin.jpg", "wisam.jpg"]

for img_path, name in zip(known_images, known_face_names):
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)

    if encodings:
        known_face_encodings.append(encodings[0])
    else:
        print(f" Warning: No face was found in the picture {img_path}")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    face_locations = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            face_locations.append((y, x + width, y + height, x))

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1

            if best_match_index != -1 and matches[best_match_index]:
                name = known_face_names[best_match_index]

        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition with Mediapipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()