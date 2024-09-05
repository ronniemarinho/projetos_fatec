import streamlit as st
import math
import networkx as nx
import folium
from streamlit_folium import st_folium

# Definição das Cidades
class Cidade:
    def __init__(self, nome, coordenadas):
        self.nome = nome
        self.coordenadas = coordenadas

# Função de heurística
def heuristica(cidade_atual, cidade_destino):
    lat1, long1 = cidade_atual.coordenadas
    lat2, long2 = cidade_destino.coordenadas
    distancia = math.sqrt((lat1 - lat2) ** 2 + (long1 - long2) ** 2)
    return distancia

# Criar o grafo com as cidades e conexões
arad = Cidade('Arad', (46.1667, 21.3167))
zerind = Cidade('Zerind', (46.6225, 21.5175))
timisoara = Cidade('Timisoara', (45.7489, 21.2087))
lugoj = Cidade('Lugoj', (45.6904, 21.9033))
mehadia = Cidade('Mehadia', (44.9000, 22.3500))
dobreta = Cidade('Dobreta', (44.7500, 22.5167))
craiova = Cidade('Craiova', (44.3167, 23.8000))
rimnicu_vilcea = Cidade('Rimnicu Vilcea', (45.1047, 24.3750))
pitesti = Cidade('Pitesti', (44.8563, 24.8696))
bucharest = Cidade('Bucharest', (44.4323, 26.1063))
sibiu = Cidade('Sibiu', (45.8035, 24.1450))
fagaras = Cidade('Fagaras', (45.8416, 24.9731))
oradea = Cidade('Oradea', (47.0485, 21.9189))

graph = nx.Graph()
graph.add_edge(arad, zerind, weight=75)
graph.add_edge(arad, timisoara, weight=118)
graph.add_edge(arad, sibiu, weight=140)
graph.add_edge(zerind, oradea, weight=71)
graph.add_edge(timisoara, lugoj, weight=111)
graph.add_edge(lugoj, mehadia, weight=70)
graph.add_edge(mehadia, dobreta, weight=75)
graph.add_edge(dobreta, craiova, weight=120)
graph.add_edge(craiova, rimnicu_vilcea, weight=146)
graph.add_edge(rimnicu_vilcea, sibiu, weight=80)
graph.add_edge(rimnicu_vilcea, pitesti, weight=97)
graph.add_edge(sibiu, fagaras, weight=99)
graph.add_edge(fagaras, bucharest, weight=211)
graph.add_edge(pitesti, bucharest, weight=101)
graph.add_edge(craiova, pitesti, weight=138)

def calcular_rota(origem, destino):
    caminho = nx.astar_path(graph, origem, destino, heuristic=heuristica, weight='weight')
    distancia = nx.astar_path_length(graph, origem, destino, heuristic=heuristica, weight='weight')
    return caminho, distancia

def main():
    st.title("Otimização de Rotas com A*")
    
    # Centralizando a imagem usando colunas
    col1, col2, col3 = st.columns([1, 2, 1])  # Definindo uma estrutura de colunas com proporções 1:2:1

    with col2:
        st.image('img.png', width=350)    # Seletor de cidades
    cidades = [arad, zerind, timisoara, lugoj, mehadia, dobreta, craiova, rimnicu_vilcea, pitesti, bucharest, sibiu, fagaras, oradea]
    cidade_opcoes = [cidade.nome for cidade in cidades]

    cidade_origem = st.selectbox("Escolha a cidade de origem", cidade_opcoes)
    cidade_destino = st.selectbox("Escolha a cidade de destino", cidade_opcoes)

    if st.button("Calcular Rota"):
        if cidade_origem and cidade_destino:
            origem = next(cidade for cidade in cidades if cidade.nome == cidade_origem)
            destino = next(cidade for cidade in cidades if cidade.nome == cidade_destino)

            # Calcular a rota
            caminho, distancia = calcular_rota(origem, destino)

            st.session_state.caminho = caminho
            st.session_state.distancia = distancia
            st.session_state.origem = origem
            st.session_state.destino = destino

    # Layout com colunas ajustadas
    col1, col2 = st.columns([5, 3])

    with col1:
        if 'caminho' in st.session_state:
            origem = st.session_state.origem
            destino = st.session_state.destino
            caminho = st.session_state.caminho
            distancia = st.session_state.distancia

            # Criar o mapa
            mapa = folium.Map(location=[44.4323, 26.1063], zoom_start=7)

            # Adicionar as cidades no mapa
            for cidade in cidades:
                folium.Marker(
                    location=[cidade.coordenadas[0], cidade.coordenadas[1]],
                    popup=cidade.nome,
                ).add_to(mapa)

            # Adicionar a rota no mapa
            rota = [(cidade.coordenadas[0], cidade.coordenadas[1]) for cidade in caminho]
            folium.PolyLine(rota, color="blue", weight=2.5, opacity=1).add_to(mapa)

            # Mostrar o mapa no Streamlit
            st_folium(mapa, width=900, height=700)

    with col2:
        if 'caminho' in st.session_state:
            origem = st.session_state.origem
            destino = st.session_state.destino
            caminho = st.session_state.caminho
            distancia = st.session_state.distancia

            st.write(f"**Caminho de {origem.nome} para {destino.nome}:**", unsafe_allow_html=True)
            st.write("**Cidades no caminho:**", unsafe_allow_html=True)
            for cidade in caminho:
                st.write(f"- **{cidade.nome}**", unsafe_allow_html=True)
            st.write(f"**Distância total: {distancia} unidades**", unsafe_allow_html=True)

    # Fórmulas
    st.subheader('Fórmulas do Algoritmo A*')

    # Fórmula da Busca A*
    st.write("**Fórmula da Busca A*:**")
    st.latex(r"G(n) = \text{custo do caminho do nó inicial até } n")
    st.latex(r"H(n) = \text{estimativa do custo do nó } n \text{ até o objetivo}")
    st.latex(r"F(n) = G(n) + H(n)")

if __name__ == '__main__':
    main()
