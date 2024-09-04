import cv2
import easyocr

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicializar EasyOCR com o idioma desejado
reader = easyocr.Reader(['en'], gpu=False)  # Especifique o idioma desejado ('en' para inglês)

while True:
    # Ler o frame da webcam
    ret, frame = cap.read()

    # Verificar se a captura de vídeo foi realizada corretamente
    if not ret:
        print("Erro ao capturar o quadro da câmera.")
        break

    # Redimensionar o frame para melhorar a performance
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Detectar e reconhecer texto na imagem
    result = reader.readtext(frame)

    # Mostrar o texto identificado na camiseta ou boné
    for (bbox, text, prob) in result:
        cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar o frame com o texto identificado
    cv2.imshow('Identificação de Texto', frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
