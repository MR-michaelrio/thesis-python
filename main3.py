from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import cv2
import os
import mysql.connector
from ultralytics import YOLO

app = FastAPI()

# Konfigurasi CORS
origins = ["https://anttendance.playandbreak.site"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO('yolov8n.pt')

MATCH_THRESHOLD = 0.8

def load_known_faces(id_company=None):
    try:
        connection = mysql.connector.connect(
            user=os.getenv("DB_USERNAME", "michael"),
            password=os.getenv("DB_PASSWORD", "Luminoso1"),
            host=os.getenv("DB_HOST", "185.199.53.230"),
            port=os.getenv("DB_PORT", "3306"),
            database=os.getenv("DB_NAME", "thesis")
        )
        cursor = connection.cursor()

        query = """
            SELECT e.id_employee, e.encoding_data 
            FROM face_encoding e
            JOIN employee emp ON e.id_employee = emp.id_employee
        """
        if id_company:
            query += " WHERE emp.id_company = %s"
            cursor.execute(query, (id_company,))
        else:
            cursor.execute(query)

        rows = cursor.fetchall()
        known_face_encodings = [np.frombuffer(row[1], dtype=np.float64) for row in rows]
        known_face_names = [row[0] for row in rows]

        cursor.close()
        connection.close()

        return known_face_encodings, known_face_names

    except Exception as e:
        print(f"Error loading known faces from the database: {e}")
        return [], []

@app.post("/process_frame")
async def process_frame(id_company: str = Form(...), image: UploadFile = File(...)):
    try:
        known_face_encodings, known_face_names = load_known_faces(id_company)

        if not known_face_encodings:
            raise HTTPException(status_code=404, detail="No known faces available.")

        img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)

        results = model(img)
        detections = results[0].boxes.xyxy.numpy()
        classes = results[0].boxes.cls.numpy()
        confidences = results[0].boxes.conf.numpy()

        person_detected = any(model.names[int(cls)] == 'person' for cls in classes)
        cellphone_detected = any(model.names[int(cls)] == 'cellphone' for cls in classes)

        if cellphone_detected:
            raise HTTPException(status_code=400, detail="Cellphone detected. Process halted.")

        if not person_detected:
            return {"message": "No person detected."}

        face_locations = face_recognition.face_locations(img, model="cnn")
        if not face_locations:
            return {"message": "No faces detected."}

        face_encodings = face_recognition.face_encodings(img, face_locations)
        employees = []

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < MATCH_THRESHOLD:
                name = known_face_names[best_match_index]

                connection = mysql.connector.connect(
                    user=os.getenv("DB_USERNAME", "michael"),
                    password=os.getenv("DB_PASSWORD", "Luminoso1"),
                    host=os.getenv("DB_HOST", "185.199.53.230"),
                    port=os.getenv("DB_PORT", "3306"),
                    database=os.getenv("DB_NAME", "thesis")
                )
                cursor = connection.cursor(dictionary=True)

                cursor.execute("""
                    SELECT e.full_name, u.identification_number
                    FROM employee e
                    LEFT JOIN users u ON e.id_users = u.id_user
                    WHERE e.id_employee = %s
                """, (name,))

                employee_data = cursor.fetchone()
                if employee_data:
                    employees.append(employee_data)

                cursor.close()
                connection.close()

        detections_list = [
            {
                'name': model.names[int(cls)],
                'box': box.tolist(),
                'confidence': float(conf)
            } for box, cls, conf in zip(detections, classes, confidences)
        ]

        return JSONResponse({
            'face_locations': face_locations,
            'face_names': known_face_names,
            'employees': employees,
            'detections': detections_list
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_face")
async def train_face(id_employee: str = Form(...), id_company: str = Form(...), image: UploadFile = File(...)):
    try:
        img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)

        face_encodings = face_recognition.face_encodings(img)
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No faces found in the image.")

        encoding = face_encodings[0].tobytes()

        connection = mysql.connector.connect(
            user=os.getenv("DB_USERNAME", "michael"),
            password=os.getenv("DB_PASSWORD", "Luminoso1"),
            host=os.getenv("DB_HOST", "185.199.53.230"),
            port=os.getenv("DB_PORT", "3306"),
            database=os.getenv("DB_NAME", "thesis")
        )
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO face_encoding (id_employee, encoding_data, id_company)
            VALUES (%s, %s, %s)
        """, (id_employee, encoding, id_company))

        connection.commit()

        cursor.close()
        connection.close()

        return {"message": "Face encoding saved successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6002)
