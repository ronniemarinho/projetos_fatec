# -*- coding: utf-8 -*-

import dlib
import math
import random
import threading
import sys
import pyttsx3

# Inicializa o mecanismo de síntese de fala
engine = pyttsx3.init()
sys.stdout.reconfigure(encoding='utf-8')


#pip install "C:\dlib-master\python_examples\dlib-19.19.0-cp38-cp38-win_amd64.whl"

def display_image(img):
    cv2.imshow('Imagem', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Carregar o modelo de detecção facial do dlib
detector = dlib.get_frontal_face_detector()

# Carregar o modelo de detecção de pontos faciais do dlib
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")



# Carregar uma rede neural treinada para estimar a idade
# Este é um exemplo simplificado, você pode precisar de um modelo mais complexo para melhores resultados
def estimate_age(face):
    # Exemplo simples: idade estimada = distância média entre os olhos * fator
    left_eye = face.part(36)
    right_eye = face.part(45)
    eye_distance = math.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)
    age = int(eye_distance * 0.1999)  # Fator de escala ajustável
    print(age)
    return age

import cv2


# Inicializa o objeto da câmera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verifica se a câmera está aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Gera a fala e reproduz
engine.say("Olá. Sou a Inteligência Artificial da Fatec Adamantina. Vamos brincar?")
engine.runAndWait()


while True:
    ret, frame = cap.read()

    # Exibe o quadro da câmera
    cv2.imshow('IA Fatec Adamantina', frame)

    # Aguarda a tecla "c" para tirar a foto
    key = cv2.waitKey(1)
    if key == ord('c'):
        # Salva a foto em um arquivo
        cv2.imwrite('foto.jpg', frame)
        print("Foto tirada e salva como 'foto.jpg'")
        break
    elif key == 27:  # Tecla "Esc" para sair
        break

# Fecha a câmera e janelas
cap.release()
cv2.destroyAllWindows()


# Carregar a imagem da pessoa
image_path = "foto.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#####
# Carregue a imagem de sobreposição (com fundo transparente)
imagem_sobreposicao = cv2.imread("adamantina.png", cv2.IMREAD_UNCHANGED)  # Substitua pelo caminho da sua imagem de sobreposição

imagem_sobreposicao = cv2.resize(imagem_sobreposicao, (150, 80))

# Defina a posição onde a imagem de sobreposição será inserida na imagem maior
posicao_x = 15  # Substitua pela posição x desejada
posicao_y = 5  # Substitua pela posição y desejada

# Obtenha as dimensões da imagem de sobreposição
altura_sobreposicao, largura_sobreposicao = imagem_sobreposicao.shape[:2]

# Calcule as coordenadas da região onde a imagem de sobreposição será inserida
y1, y2 = max(posicao_y, 0), min(posicao_y + altura_sobreposicao, image.shape[0])
x1, x2 = max(posicao_x, 0), min(posicao_x + largura_sobreposicao, image.shape[1])

# Calcule os offsets para a sobreposição
dy, dx = max(0, -posicao_y), max(0, -posicao_x)

# Extraia os canais alfa e RGB da imagem de sobreposição
canal_alpha = imagem_sobreposicao[dy:y2 - y1 + dy, dx:x2 - x1 + dx, 3]
canais_rgb = imagem_sobreposicao[dy:y2 - y1 + dy, dx:x2 - x1 + dx, :3]

# Faça a sobreposição da imagem de sobreposição na imagem maior, levando em conta o canal alfa
image[y1:y2, x1:x2][canal_alpha > 0] = canais_rgb[canal_alpha > 0]


# Salve a imagem resultante
#cv2.imwrite("imagem_resultante.png", image)

#print("Imagem com sobreposição criada e salva como 'imagem_resultante.png'")

###

# Detectar rostos na imagem
faces = detector(gray)

# Iterar sobre os rostos detectados
for face in faces:
    # Detectar pontos faciais
    landmarks = predictor(gray, face)

    # Estimar a idade
    age = estimate_age(landmarks)

    # Desenhar um retângulo ao redor do rosto
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    numero_aleatorio = random.randint(10, age)
    print("numero aleatorio",numero_aleatorio)
    # Escrever a idade na imagem
    texto= f"Idade estimada: {age} anos"
    #texto= 'Sua idade estimada é de', age,' anos, mas tem rosto de {numero_aleatorio} '.encode('utf-8').decode('utf-8')

    cv2.putText(image, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #cv2.putText(image, f"Sua idade \u00e9 {age} , com cara de {numero_aleatorio}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #print(numero_aleatorio)

# Reduzir o tamanho da caixa de impressão para a imagem menor
small_image_resized = cv2.resize(image, (1080, 720))  # Ajuste o tamanho conforme necessário



# Mostrar a imagem com as informações
#img=cv2.imshow("Idade estimada", small_image_resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Cria uma nova thread para exibir a imagem
display_thread = threading.Thread(target=display_image, args=(small_image_resized,))
display_thread.start()





# Define propriedades de voz (opcional)
engine.setProperty("rate", 190)  # Velocidade da fala (palavras por minuto)
engine.setProperty("volume", 1.0)  # Volume da fala (0.0 a 1.0)

texto=f"Você possui {age}, mas tem cara de {numero_aleatorio} anos"

# Gera a fala e reproduz
engine.say(texto)
engine.runAndWait()

# Texto que você deseja transformar em fala
if age>16 and age<20:
    texto = "Venha prestar o vestibular da Fatec Adamantina"
    # Gera a fala e reproduz
    engine.say(texto)
    engine.runAndWait()


#####################################################################
