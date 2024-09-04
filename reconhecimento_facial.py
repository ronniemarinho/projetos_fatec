import cv2

# Carregar o classificador de rosto pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Carregar a imagem que será sobreposta no canto superior esquerdo
image_path = '/Users/ronnieshida/PycharmProjects/novoprojeto/img.png'  # Substitua com o caminho da sua imagem
img_overlay = cv2.imread(image_path, -1)  # -1 para carregar a imagem com transparência (alfa channel)

# Redimensionar a imagem para um tamanho específico (opcional)
# Você pode ajustar a largura e altura conforme necessário
img_overlay = cv2.resize(img_overlay, (540, 300))  # Exemplo de redimensionamento para 100x100 pixels

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

    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Sobrepor a imagem no canto superior esquerdo do frame
    rows, cols, channels = img_overlay.shape
    roi = frame[0:rows, 0:cols]

    # Criar uma máscara de transparência para a imagem sobreposta
    img_gray = cv2.cvtColor(img_overlay[:, :, :3], cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Aplicar a máscara à região de interesse (ROI) do frame
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(img_overlay[:, :, :3], img_overlay[:, :, :3], mask=mask)
    dst = cv2.add(img_bg, img_fg)

    # Atualizar o frame com a imagem sobreposta
    frame[0:rows, 0:cols] = dst

    # Mostrar o frame com os rostos detectados e a imagem sobreposta
    cv2.imshow('Reconhecimento Facial', frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
