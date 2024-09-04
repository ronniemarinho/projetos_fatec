import cv2
import dlib
import numpy as np
from PIL import Image

# Carregar o classificador de rosto pré-treinado do dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Carregar as imagens de rosto para troca (você pode substituir com seus próprios arquivos)
img_path1 = "img_5.png"
img_path2 = "img_5.png"

# Carregar as imagens usando PIL para manter o canal alfa (transparência)
img1 = Image.open(img_path1).convert('RGBA')
img2 = Image.open(img_path2).convert('RGBA')

# Redimensionar as imagens para um tamanho comum
img1 = img1.resize((200, 200))
img2 = img2.resize((200, 200))

# Converter as imagens PIL para arrays NumPy
img_np1 = np.array(img1)
img_np2 = np.array(img2)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Ler o frame da webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = detector(gray)

    # Processar cada rosto detectado
    for face in faces:
        # Obter os marcos faciais (pontos) para o rosto detectado
        landmarks = predictor(gray, face)

        # Converter os marcos faciais para um array NumPy
        landmarks_np = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

        # Coordenadas do retângulo delimitador do rosto
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Coordenadas dos pontos do rosto
        face_points = landmarks_np[17:27]  # Selecionar pontos ao redor dos olhos
        face_hull = cv2.convexHull(face_points)  # Criar contorno convexo ao redor dos pontos do rosto

        # Extrair a região de interesse (ROI) do rosto
        roi_face = frame[y:y + h, x:x + w]

        # Dimensionar as imagens de rosto para o tamanho do rosto detectado
        img_face_resized = cv2.resize(img_np1, (w, h))
        img_mask_resized = cv2.resize(img_np2, (w, h))

        # Extrair a máscara alfa (canal alfa) das imagens de rosto
        _, alpha1 = cv2.threshold(img_face_resized[:, :, 3], 1, 255, cv2.THRESH_BINARY)
        _, alpha2 = cv2.threshold(img_mask_resized[:, :, 3], 1, 255, cv2.THRESH_BINARY)

        # Converter as imagens de rosto para o formato BGR (sem canal alfa)
        img_face_resized = img_face_resized[:, :, 0:3]
        img_mask_resized = img_mask_resized[:, :, 0:3]

        # Aplicar a máscara alfa nas imagens de rosto
        for c in range(0, 3):
            roi_face[:, :, c] = (alpha1 / 255.0) * img_face_resized[:, :, c] + \
                                (1.0 - alpha1 / 255.0) * roi_face[:, :, c]

            roi_face[:, :, c] = (alpha2 / 255.0) * img_mask_resized[:, :, c] + \
                                (1.0 - alpha2 / 255.0) * roi_face[:, :, c]

        # Sobrepor o rosto modificado de volta ao frame original
        frame[y:y + h, x:x + w] = roi_face

    # Mostrar o frame com a troca de rosto em tempo real
    cv2.imshow("Troca de Rosto em Tempo Real", frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
