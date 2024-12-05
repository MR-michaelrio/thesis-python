from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
from flask_cors import CORS
import mysql.connector
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/process_frame": {"origins": "http://127.0.0.1:8000"}})
model = YOLO('yolov8n.pt')  # Ensure the model path is correct

# Fungsi untuk memuat encoding wajah yang disimpan di database
def load_known_faces():
    try:
        connection = mysql.connector.connect(
            user=os.getenv("DB_USERNAME", "michael"),
            password=os.getenv("DB_PASSWORD", "Luminoso1"),
            host=os.getenv("DB_HOST", "185.199.53.230"),
            port=int(os.getenv("DB_PORT", "3306")),
            database=os.getenv("DB_DATABASE", "thesis")
        )
        cursor = connection.cursor()
        cursor.execute("SELECT id_employee, encoding_data, id_company FROM face_encoding")
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
        print(f"Error loading known faces: {e}")
        return [], []

# Fungsi untuk menangani proses frame (deteksi wajah, ekstraksi encoding, dan pencocokan)
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Ambil gambar dari request
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image provided'}), 400

        # Membaca gambar dan mengonversi ke format yang dibutuhkan
        img_array = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Deteksi wajah menggunakan YOLO dan face_recognition
        results = model(image)  # Menggunakan YOLO untuk mendeteksi wajah
        boxes = results.xyxy[0].cpu().numpy()  # Mendapatkan bounding box wajah yang terdeteksi
        
        face_locations = []
        for box in boxes:
            # Box: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box[:4])
            face_locations.append((y1, x2, y2, x1))  # face_recognition membutuhkan urutan (top, right, bottom, left)
        
        # Mengambil encoding wajah menggunakan face_recognition
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Memuat wajah yang sudah terdaftar di database
        known_face_encodings, known_face_names = load_known_faces()
        
        face_names = []
        for face_encoding in face_encodings:
            # Mencocokkan wajah yang terdeteksi dengan wajah yang ada di database
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

        return jsonify({'detected_faces': face_names, 'boxes': boxes.tolist()})

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': 'An error occurred during face processing'}), 500

# Fungsi untuk menyimpan encoding wajah ke database (pastikan kolom 'encoding_data' adalah LONGBLOB)
@app.route('/train_face', methods=['POST'])
def train_face():
    try:
        # Ambil data wajah (encoding) dari request
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image file provided'}), 400

        # Baca gambar dari file
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Ekstrak encoding wajah
        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) > 0:
            encoding = face_encodings[0]

            # Koneksi ke database MySQL
            connection = mysql.connector.connect(
                user=os.getenv("DB_USERNAME", "michael"),
                password=os.getenv("DB_PASSWORD", "Luminoso1"),
                host=os.getenv("DB_HOST", "185.199.53.230"),
                port=int(os.getenv("DB_PORT", "3306")),
                database=os.getenv("DB_DATABASE", "thesis")
            )
            cursor = connection.cursor()

            # Ambil ID karyawan dan perusahaan dari form request
            id_employee = request.form.get('id_employee')
            id_company = request.form.get('id_company')

            if not id_employee or not id_company:
                return jsonify({'error': 'Employee ID or Company ID is missing'}), 400

            # Simpan encoding wajah ke database
            cursor.execute(
                "INSERT INTO face_encoding (id_employee, encoding_data, id_company) VALUES (%s, %s, %s)",
                (id_employee, np.array(encoding).tobytes(), id_company)
            )

            connection.commit()
            cursor.close()
            connection.close()

            return jsonify({'status': 'success', 'message': 'Face encoding saved successfully'}), 200
        else:
            return jsonify({'error': 'No faces found in image'}), 400

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6002)
