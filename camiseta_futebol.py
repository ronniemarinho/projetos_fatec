import cv2
import numpy as np

# Carregar o classificador de rosto pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Carregar a imagem do chapéu
image_path = 'img_4.png'  # Substitua com o caminho da sua imagem
img_overlay = cv2.imread(image_path, -1)  # -1 para carregar a imagem com transparência (alfa channel)

# Redimensionar a imagem do chapéu para um tamanho maior
img_overlay = cv2.resize(img_overlay, (900, 800))  # Exemplo de redimensionamento para 300x200 pixels

while True:
    # Ler o frame da webcam
    ret, frame = cap.read()

    # Verificar se a captura de vídeo foi realizada corretamente
    if not ret:
        print("Erro ao capturar o quadro da câmera.")
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem em escala de cinza
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Processar cada rosto detectado
    for (x, y, w, h) in faces:
        # Ajustar a posição e o tamanho do chapéu para o rosto detectado
        hat_width = w
        hat_height = int(h * 0.8)  # Exemplo de altura do chapéu maior que a altura do rosto
        hat_resized = cv2.resize(img_overlay, (hat_width, hat_height))

        # Calcular as coordenadas para sobrepor o chapéu no rosto
        x_hat = x
        y_hat = y - int(h * 0.6)  # Exemplo de ajuste vertical do chapéu

        # Sobrepor o chapéu na região do rosto
        frame_alpha = frame.copy()
        roi = frame_alpha[y_hat:y_hat + hat_height, x_hat:x_hat + hat_width]

        alpha_channel = hat_resized[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = alpha_channel * hat_resized[:, :, c] + (1 - alpha_channel) * roi[:, :, c]

        frame_alpha[y_hat:y_hat + hat_height, x_hat:x_hat + hat_width] = roi

        # Mostrar o frame com os rostos detectados e o chapéu sobreposto
        cv2.imshow('Reconhecimento Facial com Chapéu', frame_alpha)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
