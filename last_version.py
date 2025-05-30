import cv2
import numpy as np
import requests
import time
import json
import os
import tensorflow as tf
from datetime import datetime, date
import threading

# Configuration
SERVER_URL = "http://localhost:3000/api"  # Replace with your server URL
CAMERA_ID = 0
FACE_DETECTION_CONFIDENCE = 0.7  # Increased to reduce false positives
RECOGNITION_THRESHOLD = 0.5      # Similar to THRESH in Kotlin code
RECOGNITION_PAD = 0.1            # Similar to PAD in Kotlin code
INPUT_SIZE = 112                 # MobileFaceNet input size
EMBEDDING_SIZE = 192             # Output embedding size
CLASS_ID = 3                     # The class ID to track attendance for
PRESENCE_THRESHOLD = 60          # Minimum seconds to consider a student present
DOOR_COOLDOWN = 5                # Seconds to wait before allowing another detection for the same student

class Student:
    def __init__(self, id, name, surname, embedding):
        self.id = id
        self.name = name
        self.surname = surname
        self.embedding = np.array(embedding, dtype=np.float32)
        self.unit_embedding = self.normalize_embedding(self.embedding)
        self.entries = []     # List of entry timestamps
        self.exits = []       # List of exit timestamps
        self.total_present_time = 0  # in seconds
        self.is_present = False
        self.last_detection_time = 0
        self.detection_count = 0  # Count of detections through the doorway
        
    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def process_detection(self):
        """Process a new detection and determine if it's an entry or exit"""
        current_time = time.time()
        
        # Check if enough time has passed since last detection (to prevent multiple counts)
        if current_time - self.last_detection_time < DOOR_COOLDOWN:
            return False, None
            
        # Update detection time
        self.last_detection_time = current_time
        
        # Increment detection count
        self.detection_count += 1
        
        # Odd count = inside (entry), Even count = outside (exit)
        if self.detection_count % 2 == 1:  # Odd: Entry
            entry_time = datetime.now()
            self.entries.append(entry_time)
            self.is_present = True
            print(f"{self.name} {self.surname} ENTERED at {entry_time}")
            return True, "entry"
        else:  # Even: Exit
            if len(self.entries) > 0:
                exit_time = datetime.now()
                self.exits.append(exit_time)
                
                # Calculate duration for this entry-exit pair
                latest_entry = self.entries[-1]
                duration = (exit_time - latest_entry).total_seconds()
                self.total_present_time += duration
                
                self.is_present = False
                print(f"{self.name} {self.surname} EXITED at {exit_time}, was present for {self.format_duration(duration)}")
                return True, "exit"
            return False, None
    
    def is_inside(self):
        """Check if student is inside based on detection count"""
        return self.detection_count % 2 == 1
    
    def is_present_enough(self):
        """Check if student has been present for enough time to be considered present for the day"""
        return self.total_present_time >= PRESENCE_THRESHOLD
        
    def format_duration(self, seconds):
        """Format seconds into a readable duration string"""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

class FaceRecognitionSystem:
    def __init__(self):
        self.students = []
        self.model = None
        self.face_detector = None
        self.active = False
        self.last_processed_time = 0  # To control processing rate
        self.processing_interval = 0.5  # Process every 0.5 seconds at most
        self.load_model()
        self.setup_face_detector()
        
    def load_model(self):
        # Load TFLite model for face embeddings
        print("Loading TFLite model...")
        model_path = 'mobilefacenet.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.model = interpreter
        print("Model loaded successfully")
        
    def setup_face_detector(self):
        # Using OpenCV's face detector
        print("Setting up face detector...")
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(face_cascade_path)
        print("Face detector ready")
        
    def fetch_students_data(self):
        try:
            print("Fetching students data from server...")
            response = requests.get(f"{SERVER_URL}/classes/{CLASS_ID}/students")
            if response.status_code == 200:
                data = response.json()
                self.students = [Student(
                    s['id'], 
                    s['name'], 
                    s['surname'], 
                    s['embedding']
                ) for s in data]
                print(f"Fetched {len(self.students)} students data")
                return True
            else:
                print(f"Failed to fetch students data: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error fetching students data: {e}")
            return False
            
    def preprocess_face(self, face_img):
        # Resize to input size
        face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
        
        # Convert to float and normalize to [-1, 1] as in the Kotlin code
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 127.5
        
        # Reshape to model input shape
        return face_img.reshape(1, INPUT_SIZE, INPUT_SIZE, 3)
    
    def generate_embedding(self, face_img):
        # Preprocess the face image
        input_data = self.preprocess_face(face_img)
        
        # Get input and output details
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        # Set input tensor
        self.model.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        self.model.invoke()
        
        # Get output tensor
        embedding = self.model.get_tensor(output_details[0]['index'])[0]
        return embedding
    
    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def cosine_distance(self, embedding1, embedding2):
        return 1.0 - np.dot(embedding1, embedding2)
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=7,  # Increased to reduce false positives
            minSize=(50, 50)  # Increased minimum face size
        )
        return faces
    
    def process_frame(self, frame):
        # Add doorway monitor indicator
        cv2.putText(frame, "DOORWAY MONITOR", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
        # Control processing rate to avoid too frequent detections
        current_time = time.time()
        should_process = current_time - self.last_processed_time >= self.processing_interval
        
        faces = self.detect_faces(frame)
        
        if len(faces) > 0:
            # Get the largest face (closest to camera)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            (x, y, w, h) = largest_face
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Only process if enough time has passed
            if should_process:
                self.last_processed_time = current_time
                
                # Crop face
                face_img = frame[y:y+h, x:x+w]
                
                # Generate embedding
                try:
                    embedding = self.generate_embedding(face_img)
                    unit_embedding = self.normalize_embedding(embedding)
                    
                    # Match with students
                    best_match = None
                    best_distance = float('inf')
                    second_best_distance = float('inf')
                    
                    for student in self.students:
                        distance = self.cosine_distance(unit_embedding, student.unit_embedding)
                        
                        if distance < best_distance:
                            second_best_distance = best_distance
                            best_distance = distance
                            best_match = student
                        elif distance < second_best_distance:
                            second_best_distance = distance
                    
                    # Check if we have a confident match
                    if best_match and best_distance < RECOGNITION_THRESHOLD and (second_best_distance - best_distance) > RECOGNITION_PAD:
                        # Draw name and status
                        status = "Inside" if best_match.is_inside() else "Outside"
                        status_color = (0, 255, 0) if best_match.is_inside() else (0, 0, 255)
                        
                        cv2.putText(frame, f"{best_match.name} {best_match.surname}", 
                                    (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"Status: {status}", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
                        
                        # Process the detection
                        success, event_type = best_match.process_detection()
                        
                        if success:
                            if event_type == "entry":
                                # Show visual confirmation of entry
                                cv2.putText(frame, "ENTRY RECORDED!", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            elif event_type == "exit":
                                # Show visual confirmation of exit
                                cv2.putText(frame, "EXIT RECORDED!", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error processing face: {e}")
        
        return frame
    
    def send_attendance_data(self):
        """Send attendance data to the server in the required format"""
        # Format as required by the server
        today_str = date.today().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
        
        attendance_list = []
        
        for student in self.students:
            # Make sure all entries have corresponding exits
            if student.is_present:
                # Force an exit at the end of the session
                exit_time = datetime.now()
                student.exits.append(exit_time)
                
                # Calculate duration for this entry-exit pair
                if len(student.entries) > len(student.exits) - 1:
                    latest_entry = student.entries[-1]
                    duration = (exit_time - latest_entry).total_seconds()
                    student.total_present_time += duration
                    
                student.is_present = False
                print(f"{student.name} {student.surname} automatically checked out at end of session")
            
            # Add to attendance list if student was present for enough time
            attendance_list.append({
                "studentId": student.id,
                "present": student.is_present_enough(),
                "duration": int(student.total_present_time/60)  # Send duration in seconds
            })
        
        if attendance_list:
            try:
                attendance_data = {
                    "date": today_str,
                    "list": attendance_list
                }
                
                print("Sending attendance data to server:")
                for item in attendance_list:
                    student = next((s for s in self.students if s.id == item["studentId"]), None)
                    if student:
                        print(f"  {student.name} {student.surname}: Present={item['present']}, Duration={self.format_duration(item['duration'])}")
                
                response = requests.post(
                    f"{SERVER_URL}/classes/{CLASS_ID}/attendances",
                    json=attendance_data
                )
                
                if response.status_code in [200, 201]:
                    print("Attendance data sent successfully")
                    
                    # Print summary
                    present_count = sum(1 for item in attendance_list if item["present"])
                    print(f"Summary: {present_count} of {len(attendance_list)} students marked present")
                    
                    return True
                else:
                    print(f"Failed to send attendance data: {response.status_code}")
                    print(f"Response: {response.text}")
                    return False
            except Exception as e:
                print(f"Error sending attendance data: {e}")
                return False
        else:
            print("No attendance data to send")
            return True
    
    def run(self):
        if not self.fetch_students_data():
            print("Failed to fetch students data. Exiting.")
            return
        
        # Start camera
        cap = cv2.VideoCapture(CAMERA_ID)
        self.active = True
        
        # Set resolution to improve performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n--- STARTING DOORWAY MONITORING ---")
        print("The system will automatically track entries and exits")
        print("- Odd detection count = Student is INSIDE")
        print("- Even detection count = Student is OUTSIDE")
        print("Press 'q' to quit and send attendance data\n")
        
        try:
            while self.active:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow('Face Recognition', processed_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            # Clean up
            self.active = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Send attendance data
            self.send_attendance_data()
            
            # Print detailed attendance
            self.print_attendance_summary()
    
    def print_attendance_summary(self):
        """Print a summary of student attendance"""
        print("\n----- ATTENDANCE SUMMARY -----")
        print(f"Date: {date.today().strftime('%Y-%m-%d')}")
        print(f"Class ID: {CLASS_ID}")
        print("--------------------------------")
        
        for student in self.students:
            present_status = "Present" if student.is_present_enough() else "Absent"
            presence_duration = self.format_duration(student.total_present_time)
            detection_count = student.detection_count
            print(f"{student.name} {student.surname}: {present_status} ({presence_duration}) - Detected {detection_count} times")
            
            # Print entry-exit pairs
            for i in range(min(len(student.entries), len(student.exits))):
                entry = student.entries[i].strftime('%H:%M:%S')
                exit = student.exits[i].strftime('%H:%M:%S')
                duration = (student.exits[i] - student.entries[i]).total_seconds()
                print(f"  Entry: {entry} - Exit: {exit} ({self.format_duration(duration)})")
            
            # Handle case where there are more entries than exits
            if len(student.entries) > len(student.exits):
                entry = student.entries[-1].strftime('%H:%M:%S')
                print(f"  Entry: {entry} - Exit: (did not exit)")
            
            print("--------------------------------")
    
    def format_duration(self, seconds):
        """Format seconds into a readable duration string"""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    face_system.run()
