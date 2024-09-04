import dlib
import cv2
import numpy as np

# Carregar o detector de faces do dlib
detector = dlib.get_frontal_face_detector()

# Carregar a imagem do chapéu de festa junina
hat_img = cv2.imread('img_1.png', cv2.IMREAD_UNCHANGED)
hat_height, hat_width, _ = hat_img.shape

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Ler um frame da webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem usando o detector do dlib
    faces = detector(gray)

    # Iterar sobre as faces detectadas
    for face in faces:
        # Coordenadas do retângulo da face
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Calcular as coordenadas para o chapéu na cabeça
        hat_top = int(y1 - 0.5 * hat_height)  # Colocar o chapéu acima da cabeça
        hat_left = int(x1 + (x2 - x1) / 2 - 0.5 * hat_width)  # Centralizar o chapéu na horizontal

        # Limitar as coordenadas para garantir que o chapéu fique dentro do frame
        hat_top = max(hat_top, 0)
        hat_left = max(hat_left, 0)
        hat_bottom = hat_top + hat_height
        hat_right = hat_left + hat_width

        # Verificar se o chapéu está dentro do frame
        if hat_bottom < frame.shape[0] and hat_right < frame.shape[1]:
            # Criar uma máscara para o chapéu
            hat_mask = hat_img[:, :, 3]  # Canal alfa da imagem do chapéu
            hat_mask_inv = cv2.bitwise_not(hat_mask)

            # Obter as regiões de interesse para o chapéu
            roi_hat = frame[hat_top:hat_top + hat_height, hat_left:hat_left + hat_width]

            # Aplicar a máscara ao chapéu e à região do frame
            hat_area = cv2.bitwise_and(hat_img[:, :, :3], hat_img[:, :, :3], mask=hat_mask)
            bg_area = cv2.bitwise_and(roi_hat, roi_hat, mask=hat_mask_inv)
            final_hat = cv2.add(hat_area, bg_area)

            # Substituir a região do chapéu no frame original
            frame[hat_top:hat_top + hat_height, hat_left:hat_left + hat_width] = final_hat

    # Exibir o frame resultante
    cv2.imshow('Festa Junina', frame)

    # Verificar se o usuário pressionou a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
