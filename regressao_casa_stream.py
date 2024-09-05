import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar base de dados
base_casas = pd.read_csv('house_prices.csv')

# Preprocessamento
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
    st.image('img.png', width=350)    # Seletor de cidades
st.write("Insira a metragem da casa para obter a previsão do preço:")

metragem = st.number_input("Metragem da casa:", min_value=0, value=0)

# Fórmula de regressão linear
st.subheader("Fórmula da Regressão Linear:")
st.latex(r'Preço = \beta_0 + \beta_1 \cdot Metragem')
st.write(f"Com os valores do modelo treinado:")
st.latex(f"Preço = {regressor_simples_casas.intercept_:.2f} + {regressor_simples_casas.coef_[0]:.2f} * Metragem")

if metragem > 0:
    # Predição para o valor inserido
    previsao = regressor_simples_casas.predict([[metragem]])[0]
    
    # Exibir o preço previsto em fonte maior e em negrito
    st.markdown(f"<h2><b>Preço previsto para uma casa com {metragem} metros quadrados: R${previsao:.2f}</b></h2>", unsafe_allow_html=True)

    # Mostrar fórmula instanciada
    st.write("Fórmula com os valores inseridos:")
    st.latex(f"Preço = {regressor_simples_casas.intercept_:.2f} + {regressor_simples_casas.coef_[0]:.2f} * {metragem}")

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
    r2 = r2_score(y_casas_teste, previsoes_teste)

    st.write(f"Erro médio absoluto (MAE): R${mae:.2f}")
    st.write(f"Erro quadrático médio (MSE): R${mse:.2f}")
    st.write(f"Raiz do erro quadrático médio (RMSE): R${rmse:.2f}")
    st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")

    # Covariância e Coeficiente de Correlação
    cov = np.cov(X_casas_treinamento.ravel(), y_casas_treinamento)[0, 1]
    cor = np.corrcoef(X_casas_treinamento.ravel(), y_casas_treinamento)[0, 1]

    st.write(f"Covariância: {cov:.2f}")
    st.write(f"Coeficiente de Correlação: {cor:.2f}")

    # Exibindo as fórmulas instanciadas de Covariância, Correlação e R²
    st.subheader("Fórmulas Instanciadas")

    # Fórmula da Covariância
    st.latex(r'\text{Cov}(X, Y) = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{n - 1}')
    st.write(f"Com os valores calculados:")
    st.latex(f"Cov(X, Y) = {cov:.2f}")

    # Fórmula do Coeficiente de Correlação
    st.latex(r'r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}')
    st.write(f"Com os valores calculados:")
    st.latex(f"r = \\frac{{{cov:.2f}}}{{{np.std(X_casas_treinamento):.2f} \\cdot {np.std(y_casas_treinamento):.2f}}} = {cor:.2f}")

    # Fórmula do Coeficiente de Determinação (R²)
    st.latex(r'R^2 = 1 - \frac{SSE}{SST}')
    st.write(f"Com os valores calculados:")
    st.latex(f"R^2 = 1 - \\frac{{{np.sum((y_casas_teste - previsoes_teste) ** 2):.2f}}}{{{np.sum((y_casas_teste - np.mean(y_casas_teste)) ** 2):.2f}}} = {r2:.2f}")

    # Tabela de interpretação do coeficiente de correlação
    st.subheader("Interpretação do Coeficiente de Correlação")
    interpretacao = pd.DataFrame({
        "Intervalo de Correlação": ["0.0 a 0.19", "0.2 a 0.39", "0.4 a 0.59", "0.6 a 0.79", "0.8 a 1.0"],
        "Interpretação": ["Muito fraca", "Fraca", "Moderada", "Forte", "Muito forte"]
    })
    
    st.table(interpretacao)

# Rodar o aplicativo: use o comando 'streamlit run app.py' no terminal
