import streamlit as st
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carregar os dados e o modelo
with open('risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

# Escalar os dados para melhorar o desempenho do KNN
scaler = StandardScaler()
X_risco_credito = scaler.fit_transform(X_risco_credito)

# Treinar o KNN com k=3 e distância euclidiana
knn_risco_credito = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_risco_credito.fit(X_risco_credito, y_risco_credito)

# Função para prever o risco de crédito usando KNN
def prever_risco_knn(historia, divida, garantias, renda):
    previsao = knn_risco_credito.predict(scaler.transform([[historia, divida, garantias, renda]]))
    return previsao[0]

# Interface do Streamlit
st.title('Classificação de Risco de Crédito com KNN')

# Centralizando a imagem usando colunas
col1, col2, col3 = st.columns([1, 2, 1])  # Definindo uma estrutura de colunas com proporções 1:2:1
with col2:
    st.image('/Users/ronnieshida/PycharmProjects/novoprojeto/img.png', width=350)

st.header('Insira os dados do cliente:')
historia = st.selectbox('História de crédito', ['Boa (0)', 'Desconhecida (1)', 'Ruim (2)'])
divida = st.selectbox('Dívida', ['Alta (0)', 'Baixa (1)'])
garantias = st.selectbox('Garantias', ['Nenhuma (0)', 'Adequada (1)'])
renda = st.selectbox('Renda', ['> 35 (2)', '15 - 35 (1)', '< 15 (0)'])

# Mapear as entradas para os valores numéricos
historia_map = {'Boa (0)': 0, 'Desconhecida (1)': 1, 'Ruim (2)': 2}
divida_map = {'Alta (0)': 0, 'Baixa (1)': 1}
garantias_map = {'Nenhuma (0)': 0, 'Adequada (1)': 1}
renda_map = {'> 35 (2)': 2, '15 - 35 (1)': 1, '< 15 (0)': 0}

# Prever quando o botão for clicado
if st.button('Prever Risco de Crédito'):
    historia_val = historia_map[historia]
    divida_val = divida_map[divida]
    garantias_val = garantias_map[garantias]
    renda_val = renda_map[renda]
    resultado = prever_risco_knn(historia_val, divida_val, garantias_val, renda_val)

    st.write('A previsão do risco de crédito é:')
    if resultado == 0:
        st.write('Baixo')
    elif resultado == 1:
        st.write('Moderado')
    else:
        st.write('Alto')

    # Mostrar a fórmula do KNN e instanciá-la com os valores inseridos
    st.subheader("Fórmula do KNN:")
    st.latex(r'D(X, Y) = \sqrt{\sum_{i=1}^{n} (X_i - Y_i)^2}')

    st.subheader("Fórmula Instanciada (Distância Euclidiana):")
    distancia_instanciada = np.sqrt((historia_val - 1)**2 + (divida_val - 1)**2 + (garantias_val - 1)**2 + (renda_val - 1)**2)
    st.latex(f'D(X, Y) = \\sqrt{{({historia_val} - 1)^2 + ({divida_val} - 1)^2 + ({garantias_val} - 1)^2 + ({renda_val} - 1)^2}} = {distancia_instanciada:.2f}')

# Exibir as classes do modelo
st.subheader('Informações do Modelo')
st.write('Classes do modelo:')
st.write(knn_risco_credito.classes_)
