import cv2
import dlib
import math

# Carregar o detector de faces do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar o modelo pré-treinado de detecção de pontos fiduciais (shape_predictor_68_face_landmarks.dat)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Função para estimar a idade com base na distância entre os olhos
def estimate_age(eye_distance):
    # A fórmula a seguir é apenas um exemplo simples e pode não ser muito precisa
    # Você pode ajustar os valores de acordo com seu próprio modelo
    age = math.exp(eye_distance * 0.) * 5
    return age


# Carregar a imagem
image = cv2.imread("foto.jpg")

# Converter a imagem em escala de cinza para a detecção facial
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostos na imagem
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Para cada rosto detectado, estimar a idade
for (x, y, w, h) in faces:
    # Detectar pontos fiduciais faciais
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    shape = predictor(gray, rect)

    # Calcular a distância entre os olhos (exemplo de uso)
    left_eye = shape.part(36)
    right_eye = shape.part(45)
    eye_distance = math.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)

    # Estimar a idade com base na distância entre os olhos
    estimated_age = estimate_age(eye_distance)

    # Desenhar um retângulo e exibir a idade estimada
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f'Age: {int(estimated_age)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Exibir a imagem com as estimativas de idade
cv2.imshow("Age Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
