import dlib
import math
import random
import threading
import sys
import pyttsx3
import cv2
import numpy as np
import re

# Model paths
AGE_MODEL = 'deploy_age.prototxt'
AGE_PROTO = 'age_net.caffemodel'
FACE_PROTO = 'deploy.prototxt.txt'
FACE_MODEL = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

# Age intervals
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Initialize frame size
frame_width = 1280
frame_height = 720

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

# Load facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

def estimate_age(face):
    # Simple example: estimated age = average distance between eyes * factor
    left_eye = face.part(36)
    right_eye = face.part(45)
    eye_distance = math.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)
    age_estimate = int(eye_distance * 0.187)  # Adjustable scale factor
    return age_estimate

def get_faces(frame, confidence_threshold=0.5):
    """Returns the box coordinates of all detected faces"""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def display_img(title, img):
    """Displays an image on screen and maintains the output until the user presses a key"""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_age(input_path: str):
    """Predict the age of the faces in the image"""
    img = cv2.imread(input_path)
    overlay_img = cv2.imread("adamantina.png", cv2.IMREAD_UNCHANGED)
    overlay_img = cv2.resize(overlay_img, (540, 300))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        age_estimate = estimate_age(landmarks)

        # Predict age using dnn
        face_img = img[face.top(): face.bottom(), face.left(): face.right()]
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False
        )
        age_net.setInput(blob)
        age_preds = age_net.forward()
        max_confidence_index = age_preds[0].argmax()

        age_interval = AGE_INTERVALS[max_confidence_index]
        match = re.search(r'\((\d+), (\d+)\)', age_interval)
        age_range = match.groups() if match else (None, None)
        min_age = int(age_range[0]) if age_range[0] else None
        max_age = int(age_range[1]) if age_range[1] else None

        age_confidence = age_preds[0][max_confidence_index] * 100
        print(f"Idade estimada: {age_estimate}, Intervalo de idade: {age_interval}, Confiança: {age_confidence:.2f}%")

        # Draw text and rectangle around the face
        label = f"Idade real entre {min_age} a {max_age} anos. Probabilidade de acerto: {age_confidence:.2f}%"
        cv2.putText(img, label, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), thickness=2)
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), thickness=2)

    # Overlay image
    pos_x, pos_y = 0, 0
    img_h, img_w = overlay_img.shape[:2]
    alpha_mask = overlay_img[:, :, 3] / 255.0
    img[pos_y:pos_y + img_h, pos_x:pos_x + img_w, 0] = (1.0 - alpha_mask) * img[pos_y:pos_y + img_h, pos_x:pos_x + img_w, 0] + alpha_mask * overlay_img[:, :, 0]
    img[pos_y:pos_y + img_h, pos_x:pos_x + img_w, 1] = (1.0 - alpha_mask) * img[pos_y:pos_y + img_h, pos_x:pos_x + img_w, 1] + alpha_mask * overlay_img[:, :, 1]
    img[pos_y:pos_y + img_h, pos_x:pos_x + img_w, 2] = (1.0 - alpha_mask) * img[pos_y:pos_y + img_h, pos_x:pos_x + img_w, 2] + alpha_mask * overlay_img[:, :, 2]

    # Display image with overlaid text and rectangles
    display_img('IA Fatec Adamantina', img)

    # Text-to-speech engine initialization
    engine = pyttsx3.init()
    engine.setProperty("rate", 190)
    engine.setProperty("volume", 1.0)

    if min_age == 15 and max_age == 20:
        engine.say("Venha prestar o vestibular da Fatec Adamantina.")

    engine.runAndWait()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        exit()

    while True:
        while True:
            ret, frame = cap.read()
            cv2.imshow('IA Fatec Adamantina', frame)

            key = cv2.waitKey(1)
            if key == ord('c'):
                cv2.imwrite('foto_fatec.jpg', frame)
                print("Foto tirada e salva como 'foto_fatec.jpg'")
                break
            elif key == 27:
                exit()

        cv2.destroyAllWindows()
        image_path = "foto_fatec.jpg"
        predict_age(image_path)
