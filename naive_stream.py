import streamlit as st
import pickle
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Carregar os dados e o modelo
with open('risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, y_risco_credito)

# Função para prever o risco de crédito
def prever_risco(historia, divida, garantias, renda):
    previsao = naive_risco_credito.predict([[historia, divida, garantias, renda]])
    return previsao[0]

# Interface do Streamlit
st.title('Classificação de Risco de Crédito')

# Centralizando a imagem usando colunas
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image('img.png', width=350)

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
    resultado = prever_risco(historia_val, divida_val, garantias_val, renda_val)

    st.write('A previsão do risco de crédito é:')
    if resultado == 0:
        st.write('Baixo')
    elif resultado == 1:
        st.write('Moderado')
    else:
        st.write('Alto')

    # Exibir a fórmula matemática do Naive Bayes
    st.subheader('Fórmula Matemática - Naive Bayes')
    st.latex(r'''
    P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
    ''')
    
    # Exibir a fórmula instanciada com os valores fornecidos
    st.subheader('Fórmula Instanciada')

    # Cálculo das probabilidades para cada classe
    n_classes = len(naive_risco_credito.classes_)
    probs = np.zeros(n_classes)

    for i in range(n_classes):
        # Probabilidade da classe
        prob_class = naive_risco_credito.class_prior_[i]

        # Probabilidade das características dado a classe
        prob_features_given_class = 1.0
        for j in range(X_risco_credito.shape[1]):
            mean = naive_risco_credito.theta_[i, j]
            var = naive_risco_credito.sigma_[i, j]
            feature_val = [historia_val, divida_val, garantias_val, renda_val][j]
            prob_feature = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((feature_val - mean) ** 2 / var))
            prob_features_given_class *= prob_feature
        
        # Probabilidade total para a classe
        probs[i] = prob_class * prob_features_given_class

    total_prob = np.sum(probs)
    probs_normalized = probs / total_prob

    st.latex(rf'''
    P(C_i|X) = \frac{P(X|C_i) \cdot P(C_i)}{P(X)}
    ''')

    for i, prob in enumerate(probs_normalized):
        st.write(f'Probabilidade para a classe {naive_risco_credito.classes_[i]}: {prob:.5f}')

# Exibir as classes do modelo
st.subheader('Informações do Modelo')
st.write('Classes do modelo:')
st.write(naive_risco_credito.classes_)
