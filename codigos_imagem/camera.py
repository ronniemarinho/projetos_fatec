import cv2
import tensorflow as tf
import numpy as np

# Carrega o modelo FER pré-treinado
model = tf.keras.models.load_model('fer_model.h5')

# Carrega uma imagem
image = cv2.imread('dora2.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Redimensiona a imagem para o tamanho esperado pelo modelo (48x48 pixels)
image_resized = cv2.resize(image_gray, (48, 48))
image_resized = np.expand_dims(image_resized, axis=0)
image_resized = np.expand_dims(image_resized, axis=-1)

# Realiza a predição usando o modelo
emotion_scores = model.predict(image_resized)
emotion_label = np.argmax(emotion_scores)

# Mapeia o resultado para as emoções
emotion_mapping = {0: "Raiva", 1: "Nojo", 2: "Medo", 3: "Feliz", 4: "Triste", 5: "Surpreso", 6: "Neutro"}
emotion_result = emotion_mapping[emotion_label]

# Exibe a emoção detectada na imagem
cv2.putText(image, emotion_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
image= cv2.resize(image, (920, 720))

cv2.imshow('Detecção de Emoção', image)

# Aguarda por uma tecla e fecha a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
