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
    
    # Cálculo dos valores instanciados (simplificado para a explicação)
    P_historia = np.exp(-0.5 * ((historia_val - naive_risco_credito.theta_[0][0]) ** 2) / naive_risco_credito.sigma_[0][0])
    P_divida = np.exp(-0.5 * ((divida_val - naive_risco_credito.theta_[0][1]) ** 2) / naive_risco_credito.sigma_[0][1])
    P_garantias = np.exp(-0.5 * ((garantias_val - naive_risco_credito.theta_[0][2]) ** 2) / naive_risco_credito.sigma_[0][2])
    P_renda = np.exp(-0.5 * ((renda_val - naive_risco_credito.theta_[0][3]) ** 2) / naive_risco_credito.sigma_[0][3])

    # Probabilidade condicional (simplificado para o exemplo)
    P_X_given_C = P_historia * P_divida * P_garantias * P_renda
    P_C = naive_risco_credito.class_prior_[0]
    P_X = P_X_given_C * P_C

    st.latex(rf'''
    P(C|X) = \frac{{P({historia_val}, {divida_val}, {garantias_val}, {renda_val}|C) \cdot P(C)}}{{P(X)}} = {P_X:.5f}
    ''')

# Exibir as classes do modelo
st.subheader('Informações do Modelo')
st.write('Classes do modelo:')
st.write(naive_risco_credito.classes_)
