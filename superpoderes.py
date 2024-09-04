import cv2
import numpy as np
import pyqrcode
import requests
import streamlit as st
import dlib
import math
import re
import time

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
    """Estimate age based on facial landmarks"""
    left_eye = face.part(36)
    right_eye = face.part(45)
    eye_distance = math.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)
    age_estimate = int(eye_distance * 0.187)  # Adjustable scale factor
    return age_estimate

def get_faces(frame, confidence_threshold=0.5):
    """Detect faces in the image"""
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

def predict_age(img):
    """Predict the age of the faces in the image"""
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

    return img

def save_and_display_image(img):
    """Save the image locally and generate a QR Code with the link to download the image"""
    cv2.imwrite('foto_fatec.jpg', img)

    # Upload the image to file.io
    with open('foto_fatec.jpg', 'rb') as f:
        response = requests.post('https://file.io/', files={'file': f})
        upload_link = response.json()['link']

    # Generate QR Code with the link to download the image
    qr = pyqrcode.create(upload_link)
    qr.png('qrcode.png', scale=10)  # Save QR Code as a PNG image

    # Read the QR Code image
    qr_image = cv2.imread('qrcode.png')

    # Resize QR Code to match the height of the captured image
    qr_height, qr_width, _ = qr_image.shape
    frame_height, frame_width, _ = img.shape
    qr_resized = cv2.resize(qr_image, (int(qr_width * frame_height / qr_height), frame_height))

    # Concatenate the captured image with the QR Code
    img_with_qr = np.hstack((img, qr_resized))

    return img_with_qr

def main():
    st.title('IA Fatec Adamantina - Captura de Foto')

    st.sidebar.header('Captura de Foto')

    if st.button('Capturar Foto - Iniciar', key='Capturar Foto - Iniciar'):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('Erro ao abrir a câmera. Certifique-se de que a câmera está conectada e tente novamente.')
            return

        # Esperar até que a câmera esteja aberta por 3 segundos
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()

        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error('Erro ao capturar imagem da câmera.')
                break

            frame = cv2.flip(frame, 1)
            stframe.image(frame, channels='BGR', caption='Câmera')

            if st.button('Capturar Foto - Confirmar', key='Capturar Foto - Confirmar'):
                img_processed = predict_age(frame)
                img_with_qr = save_and_display_image(img_processed)

                # Display processed image with QR code
                st.image(img_with_qr, channels='BGR', caption='Imagem Processada')
                break

        cap.release()

if __name__ == '__main__':
    main()
