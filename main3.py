from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
from flask_cors import CORS
import os
from ultralytics import YOLO
import mysql.connector


app = Flask(__name__)
CORS(app, resources={r"/process_frame": {"origins": "https://michael.playandbreak.site/"}})
model = YOLO('yolov8n.pt')  # Ensure the model path is correct

def load_known_faces():
    try:
        connection = mysql.connector.connect(
            user=os.getenv("DB_USERNAME", "michael"),
            password=os.getenv("DB_PASSWORD", "Luminoso1"),
            host=os.getenv("DB_HOST", "185.199.53.230"),
            port=os.getenv("DB_PORT", "3306"),
            database=os.getenv("DB_NAME", "thesis")
        )
        cursor = connection.cursor()
        cursor.execute("SELECT id_employee, encoding_data FROM face_encoding")
        rows = cursor.fetchall()
        known_face_encodings = []
        known_face_names = []

        for row in rows:
            id_employee, encoding_data = row
            known_face_encodings.append(np.frombuffer(encoding_data, dtype=np.float64))
            known_face_names.append(id_employee)

        cursor.close()
        connection.close()

        return known_face_encodings, known_face_names

    except Exception as e:
        print(f"Error loading known faces from the database: {e}")
        return [], []

# Load the known faces into memory
known_face_encodings, known_face_names = load_known_faces()
MATCH_THRESHOLD = 0.8

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Perform YOLOv8 detection (assuming this works fine)
        results = model(img)
        detections = results[0].boxes.xyxy.numpy()
        classes = results[0].boxes.cls.numpy()
        confidences = results[0].boxes.conf.numpy()

        person_detected = False
        low_confidence_person = False
        cellphone_detected = False

        for box, cls, conf in zip(detections, classes, confidences):
            class_name = model.names[int(cls)]
            if class_name == 'cellphone':
                cellphone_detected = True
            if class_name == 'person':
                person_detected = True
                if conf < 0.5:
                    low_confidence_person = True

        if low_confidence_person and cellphone_detected:
            return jsonify({'error': 'Low-confidence person detected along with a cellphone. Process halted.'}), 400

        face_locations = []
        face_names = []
        employees = []
        if person_detected and not low_confidence_person:
            face_locations = face_recognition.face_locations(img)

            if not face_locations:
                print("No faces detected.")
                return jsonify({'message': 'No faces detected in the image.'}), 200

            face_encodings = face_recognition.face_encodings(img, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)  # Get the index of the closest match

                # If the distance is below a certain threshold, use the best match
                if face_distances[best_match_index] < MATCH_THRESHOLD:
                    name = known_face_names[best_match_index]
                    connection = mysql.connector.connect(
                        host=os.getenv("DB_HOST", "185.199.53.230"),
                        port=os.getenv("DB_PORT", "3306"),
                        database=os.getenv("DB_NAME", "thesis"),
                        user=os.getenv("DB_USERNAME", "michael"),
                        password=os.getenv("DB_PASSWORD", "Luminoso1")
                    )
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM employee WHERE id_employee = %s", (name,))
                    employee_data = cursor.fetchone()  # Fetch the employee details

                    cursor.close()
                    connection.close()

                    if employee_data:
                        employees.append(employee_data)
                    print(f"Face match found: {name}")
                else:
                    print("No match found for this face.")

                face_names.append(name)

        return jsonify({
            'face_locations': face_locations,
            'face_names': face_names,
            'employees': employees,  # Add detailed employee data
            'detections': [
                {
                    'name': model.names[int(cls)],
                    'box': box.tolist(),
                    'confidence': float(conf)
                } for box, cls, conf in zip(detections, classes, confidences)
            ]
        })

    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_face', methods=['POST'])
def train_face():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400

        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) > 0:
            encoding = face_encodings[0]
            # Save the encoding to the database (MySQL)
            connection = mysql.connector.connect(
                host=os.getenv("DB_HOST", "185.199.53.230"),
                port=os.getenv("DB_PORT", "3306"),
                database=os.getenv("DB_DATABASE", "thesis"),
                user=os.getenv("DB_USERNAME", "michael"),
                password=os.getenv("DB_PASSWORD", "Luminoso1")
            )
            cursor = connection.cursor()

            # Example of inserting a new face encoding into the database
            insert_query = "INSERT INTO face_encoding (id_employee, encoding_data,id_company) VALUES (%s, %s, %s)"
            encoding_bytes = encoding.tobytes()  # Convert encoding to bytes for storage

            cursor.execute(insert_query, (request.form['id_employee'], encoding_bytes,request.form['id_company']))
            connection.commit()

            cursor.close()
            connection.close()

            return jsonify({'message': 'Face encoding saved successfully'}), 200
        else:
            return jsonify({'error': 'No faces found in the image'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6002)
