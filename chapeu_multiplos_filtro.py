import cv2
import numpy as np
import pyqrcode
import requests

# Carregar o classificador de rosto pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Carregar a imagem do chapéu
image_path = 'chapeu2.png'  # Substitua com o caminho da sua imagem
img_overlay = cv2.imread(image_path, -1)  # -1 para carregar a imagem com transparência (alfa channel)

# Redimensionar a imagem do chapéu para um tamanho maior
img_overlay = cv2.resize(img_overlay, (800, 800))  # Exemplo de redimensionamento para 800x800 pixels

# Inicializar a variável key
key = None

# Carregar a imagem que será sobreposta no canto superior esquerdo
image_path1 = '/Users/ronnieshida/PycharmProjects/novoprojeto/img.png'  # Substitua com o caminho da sua imagem
img_overlay1 = cv2.imread(image_path1, -1)  # -1 para carregar a imagem com transparência (alfa channel)

# Redimensionar a imagem para um tamanho específico (opcional)
img_overlay1 = cv2.resize(img_overlay1, (540, 300))  # Exemplo de redimensionamento para 540x300 pixels

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

    # Criar uma cópia do frame original para sobrepor os chapéus
    frame_alpha = frame.copy()

    # Processar cada rosto detectado
    for (x, y, w, h) in faces:
        # Ajustar a posição e o tamanho do chapéu para o rosto detectado
        hat_width = w
        hat_height = int(h * 1.2)  # Exemplo de altura do chapéu maior que a altura do rosto
        hat_resized = cv2.resize(img_overlay, (hat_width, hat_height))

        # Calcular as coordenadas para sobrepor o chapéu no rosto
        x_hat = x
        y_hat = y - int(h * 1.0)  # Exemplo de ajuste vertical do chapéu

        # Verificar se o chapéu cabe dentro do frame
        if y_hat >= 0 and y_hat + hat_height <= frame.shape[0] and x_hat + hat_width <= frame.shape[1]:
            # Criar uma ROI (Região de Interesse) no frame
            roi = frame_alpha[y_hat:y_hat + hat_height, x_hat:x_hat + hat_width]

            # Criar uma máscara de transparência para o chapéu
            alpha_channel = hat_resized[:, :, 3] / 255.0
            for c in range(0, 3):
                roi[:, :, c] = alpha_channel * hat_resized[:, :, c] + (1 - alpha_channel) * roi[:, :, c]

            # Substituir a ROI pelo chapéu sobreposto
            frame_alpha[y_hat:y_hat + hat_height, x_hat:x_hat + hat_width] = roi

    # Sobrepor a imagem no canto superior esquerdo do frame
    rows1, cols1, channels1 = img_overlay1.shape
    roi1 = frame_alpha[0:rows1, 0:cols1]

    # Criar uma máscara de transparência para a imagem sobreposta
    img_gray1 = cv2.cvtColor(img_overlay1[:, :, :3], cv2.COLOR_BGR2GRAY)
    ret1, mask1 = cv2.threshold(img_gray1, 10, 255, cv2.THRESH_BINARY)
    mask_inv1 = cv2.bitwise_not(mask1)

    # Aplicar a máscara à região de interesse (ROI) do frame
    img_bg1 = cv2.bitwise_and(roi1, roi1, mask=mask_inv1)
    img_fg1 = cv2.bitwise_and(img_overlay1[:, :, :3], img_overlay1[:, :, :3], mask=mask1)
    dst1 = cv2.add(img_bg1, img_fg1)

    # Atualizar o frame com a imagem sobreposta
    frame_alpha[0:rows1, 0:cols1] = dst1

    # Mostrar o frame com os rostos detectados e os chapéus sobrepostos
    cv2.imshow('Reconhecimento Facial com Chapéus', frame_alpha)

    # Tirar uma foto quando a tecla 'c' for pressionada
    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        # Salvar a foto em disco
        cv2.imwrite('foto.jpg', frame_alpha)

        # Upload da imagem para o file.io
        with open('foto.jpg', 'rb') as f:
            response = requests.post('https://file.io/', files={'file': f})
            upload_link = response.json()['link']

        # Gerar o QR Code com o link para baixar a foto
        qr = pyqrcode.create(upload_link)
        qr.png('qrcode.png', scale=10)  # Salvar o QR Code como uma imagem PNG

        # Ler a imagem do QR Code
        qr_image = cv2.imread('qrcode.png')

        # Redimensionar o QR Code para ter a mesma altura que a imagem capturada
        qr_height, qr_width, _ = qr_image.shape
        frame_height, frame_width, _ = frame.shape
        qr_resized = cv2.resize(qr_image, (int(qr_width * frame_height / qr_height), frame_height))

        # Concatenar a imagem capturada com o QR Code
        frame_with_qr = np.hstack((frame_alpha, qr_resized))

        # Apresentar a foto com o QR Code
        cv2.imshow('Foto com QR Code', frame_with_qr)

    # Verificar se uma tecla foi pressionada
    if key is not None and key & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
