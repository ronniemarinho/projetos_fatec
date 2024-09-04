import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregando a base de dados
base_casas = pd.read_csv('house_prices.csv')

# Preprocessamento dos dados
X_casas = base_casas.iloc[:, 5:6].values
y_casas = base_casas.iloc[:, 2].values

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas,
                                                                                          test_size=0.3, random_state=0)

# Treinamento do modelo de regressão linear simples
regressor_simples_casas = LinearRegression()
regressor_simples_casas.fit(X_casas_treinamento, y_casas_treinamento)

# Interface do Streamlit
st.title("Predição de Preços de Casas")
# Centralizando a imagem usando colunas
col1, col2, col3 = st.columns([1, 2, 1])  # Definindo uma estrutura de colunas com proporções 1:2:1

with col2:
    st.image('/Users/ronnieshida/PycharmProjects/novoprojeto/img.png', width=350)
#
st.write("Insira a metragem da casa para obter a previsão do preço:")

metragem = st.number_input("Metragem da casa :", min_value=0, value=0)

if metragem > 0:
    # Predição para o valor inserido
    previsao = regressor_simples_casas.predict([[metragem]])[0]
    st.write(f"Preço previsto para uma casa com {metragem} : ${previsao:.2f}")

    # Gráficos de dispersão e linha de regressão
    previsoes_treinamento = regressor_simples_casas.predict(X_casas_treinamento)
    grafico1 = px.scatter(x=X_casas_treinamento.ravel(), y=y_casas_treinamento,
                          labels={'x': 'Metragem ', 'y': 'Preço (R$)'})
    grafico2 = px.line(x=X_casas_treinamento.ravel(), y=previsoes_treinamento,
                       labels={'x': 'Metragem ', 'y': 'Preço (R$)'})
    grafico2.data[0].line.color = 'red'
    grafico = go.Figure(data=grafico1.data + grafico2.data)
    st.plotly_chart(grafico)

    # Métricas de erro
    previsoes_teste = regressor_simples_casas.predict(X_casas_teste)
    mae = mean_absolute_error(y_casas_teste, previsoes_teste)
    mse = mean_squared_error(y_casas_teste, previsoes_teste)
    rmse = np.sqrt(mse)

    st.write(f"Erro médio absoluto (MAE): ${mae:.2f}")
    st.write(f"Erro quadrático médio (MSE): ${mse:.2f}")
    st.write(f"Raiz do erro quadrático médio (RMSE): ${rmse:.2f}")

# Rodar o aplicativo: use o comando 'streamlit run app.py' no terminal
