import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Carregar os dados (simulando um arquivo CSV local)
df_comentarios = pd.read_csv('tweet2.csv')

# Inicializar o analisador de sentimentos
analisador = SentimentIntensityAnalyzer()

# Listas para armazenar comentários positivos, negativos e neutros
comentarios_positivos = []
comentarios_negativos = []
comentarios_neutros = []

# Iterar sobre cada comentário da base de dados
for comentario in df_comentarios['Comentario']:
    # Calcular a polaridade do sentimento
    polaridade = analisador.polarity_scores(comentario)['compound']
    if polaridade > 0.05:
        comentarios_positivos.append(comentario)
    elif polaridade < -0.05:
        comentarios_negativos.append(comentario)
    else:
        comentarios_neutros.append(comentario)

# Contagem de comentários
total_positivos = len(comentarios_positivos)
total_negativos = len(comentarios_negativos)
total_neutros = len(comentarios_neutros)


# Função para exibir os gráficos e comentários
def exibir_pagina():

    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Análise de Sentimentos em Comentários extraídos do Tweeter</h1>
        </div>
        """, unsafe_allow_html=True
    )
    # Centralizando a imagem usando colunas
    col1, col2, col3 = st.columns([1, 2, 1])  # Definindo uma estrutura de colunas com proporções 1:2:1

    with col2:
        st.image('/novoprojeto/img.png', width=350)
    #
    # Gráfico de pizza
    st.header('Distribuição dos Sentimentos')
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.pie([total_positivos, total_negativos, total_neutros], labels=['Positivos', 'Negativos', 'Neutros'],
            colors=['lightgreen', 'red', 'lightskyblue'], autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Gráfico de barras
    st.header('Quantidade de Comentários por Sentimento')
    fig2, ax2 = plt.subplots()
    ax2.bar(['Positivos', 'Negativos', 'Neutros'], [total_positivos, total_negativos, total_neutros],
            color=['lightgreen', 'red', 'lightskyblue'])
    ax2.set_ylabel('Quantidade')
    st.pyplot(fig2)

    # Paginação dos comentários positivos
    st.header('Comentários Positivos')
    exibir_comentarios(comentarios_positivos)

    # Paginação dos comentários negativos
    st.header('Comentários Negativos')
    exibir_comentarios(comentarios_negativos)

    # Paginação dos comentários neutros
    st.header('Comentários Neutros')
    exibir_comentarios(comentarios_neutros)


def exibir_comentarios(lista_comentarios):
    page_size = 5
    num_pages = int(len(lista_comentarios) / page_size) + 1
    page_number = st.number_input('Número da página', min_value=1, max_value=num_pages, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = min(page_number * page_size, len(lista_comentarios))

    for comentario in lista_comentarios[start_idx:end_idx]:
        st.write(comentario)


# Executar a página
if __name__ == '__main__':
    exibir_pagina()
