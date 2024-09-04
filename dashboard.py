import streamlit as st
import pandas as pd
import plotly.express as px

# Configurar o layout da página
st.set_page_config(layout="wide")

# Carregar os dados do arquivo "vendas.csv" (certifique-se de que o arquivo está no mesmo diretório)
df = pd.read_csv("vendas.csv", sep=";", decimal=",")

# Converter a coluna "Date" para o formato de data
df["Data"] = pd.to_datetime(df["Data"])
df = df.sort_values("Data")

# Criar uma nova coluna "Month" que contém o ano e o mês
df["Mês"] = df["Data"].apply(lambda x: str(x.year) + "-" + str(x.month))

# Criar uma seleção de meses na barra lateral do dashboard
month = st.sidebar.selectbox("Mês", df["Mês"].unique())
df_filtered = df[df["Mês"] == month]

# Adicionar filtro por gênero com check
generos = st.sidebar.multiselect("Gênero", df["Gênero"].unique(), default=df["Gênero"].unique())
# Título do aplicativo
#st.sidebar.title("Meu Aplicativo com Streamlit")

# Inserir a imagem
st.sidebar.image("img.png", use_column_width=True)

# Inserir o slogan
#st.sidebar.write("Ajudando você a visualizar seus dados de forma fácil e eficaz!")
if generos:
    df_filtered = df_filtered[df_filtered["Gênero"].isin(generos)]

# Adicionar cartões de dashboard
st.title("Dashboard Fatec Adamantina")
# Inserir a imagem
#st.image("img.png", caption="Logotipo da Empresa", use_column_width=True)

# Inserir o slogan
#st.write("Ajudando você a visualizar seus dados de forma fácil e eficaz!")
st.write("Prof. Dr. Ronnie Shida Marinho")


st.markdown("## Resumo")
total_faturamento = df_filtered["Total"].sum()
total_vendas = df_filtered.shape[0]
avaliacao_media = df_filtered["Rating"].mean()
total_produtos = df_filtered["Quantidade"].sum()

# Calculando ROI e CAC
custo_campanha = 5000  # Custo total da campanha
clientes_adquiridos = 1000  # Número de clientes adquiridos através da campanha
roi = (total_faturamento - custo_campanha) / custo_campanha
cac = custo_campanha / clientes_adquiridos
# Calcular o Valor Médio do Pedido (Average Order Value - AOV)
average_order_value = total_faturamento / total_vendas
# Calcular a Taxa de Conversão
taxa_conversao = total_vendas / clientes_adquiridos

# Dividir a tela em colunas para os cartões de dashboard
col1, col2, col3, col4 = st.columns(4)

# Adicionar os cartões de dashboard em cada coluna
with col1:
    st.metric(label="ROI", value=f"{roi:.2f}")

with col2:
    st.metric(label="CAC", value=f"{cac:.2f}")

with col3:
    st.metric(label="Taxa de Conversão", value=f"{taxa_conversao:.2%}")

with col4:
    st.metric(label="AOV", value=f"R${average_order_value:.2f}")

# Dividir a tela em colunas para os cartões de dashboard da segunda linha
col5, col6, col7, col8 = st.columns(4)

# Adicionar os cartões de dashboard em cada coluna da segunda linha
with col5:
    st.metric(label="Total Faturamento", value=f"R${total_faturamento:.2f}")

with col6:
    st.metric(label="Total Vendas", value=total_vendas)

with col7:
    st.metric(label="Total Produtos Vendidos", value=total_produtos)

with col8:
    st.metric(label="Avaliação Média", value=f"{avaliacao_media:.2f}")

# Dividir a tela em colunas para os gráficos
col1, col2 = st.columns(2)
col3, col4, col5 = st.columns(3)

# Criar o gráfico de faturamento por dia
fig_date = px.bar(df_filtered, x="Data", y="Total", color="Cidade", title="Faturamento por dia")
col1.plotly_chart(fig_date, use_container_width=True)

# Criar o gráfico de faturamento por tipo de produto
fig_prod = px.bar(df_filtered, x="Data", y="Linha de produto", color="Cidade", title="Faturamento por tipo de produto", orientation="h")
col2.plotly_chart(fig_prod, use_container_width=True)

# Calcular o faturamento total por cidade
city_total = df_filtered.groupby("Cidade")[["Total"]].sum().reset_index()

# Criar o gráfico de barras para exibir o faturamento por cidade
fig_city = px.bar(city_total, x="Cidade", y="Total", title="Faturamento por cidade")
col3.plotly_chart(fig_city, use_container_width=True)

# Criar o gráfico de pizza para exibir o faturamento por tipo de pagamento
fig_kind = px.pie(df_filtered, values="Total", names="Pagamento", title="Faturamento por tipo de pagamento")
col4.plotly_chart(fig_kind, use_container_width=True)

# Calcular a avaliação média por cidade
city_total = df_filtered.groupby("Cidade")[["Rating"]].mean().reset_index()

# Criar o gráfico de barras para exibir a avaliação média
fig_rating = px.bar(df_filtered, y="Rating", x="Cidade", title="Avaliação Média")
col5.plotly_chart(fig_rating, use_container_width=True)


# Calcular o total de vendas por mês e por linha de produto
vendas_por_produto_e_mes = df_filtered.groupby(["Mês", "Linha de produto"])[["Quantidade"]].sum().reset_index()

# Criar o gráfico de barras para exibir o total de vendas por mês e por linha de produto
fig_vendas_por_produto_e_mes = px.bar(vendas_por_produto_e_mes, x="Linha de produto", y="Quantidade", color="Mês",
                                      title="Total de Vendas por Mês e por Linha de Produto", facet_row="Mês",
                                      height=600, width=1200)

# Exibir o gráfico
st.plotly_chart(fig_vendas_por_produto_e_mes, use_container_width=True)
