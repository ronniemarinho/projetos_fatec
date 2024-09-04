import math
import networkx as nx
import folium
import webbrowser

class Cidade:
    def __init__(self, nome, coordenadas):
        self.nome = nome
        self.coordenadas = coordenadas

def heuristica(cidade_atual, cidade_destino):

    lat1, long1 = cidade_atual.coordenadas
    lat2, long2 = cidade_destino.coordenadas
#distancia euclidiana
    distancia = math.sqrt((lat1 - lat2)**2 + (long1 - long2)**2)

    return distancia

#Criar um grafo com as cidades
arad = Cidade('Arad',(46.1667, 21.3167))
zerind = Cidade('Zerind', (46.6225, 21.5175))
timisoara = Cidade('Timisoara', (45.7489, 21.2087))
lugoj = Cidade('Lugoj', (45.6904, 21.9033))
mehadia = Cidade('Mehadia', (44.9000, 22.3500))
dobreta = Cidade('Dobreta', (44.7500, 22.5167))
craiova = Cidade('Craiova',(44.3167, 23.8000))
rimnicu_vilcea = Cidade('Rimnicu Vilcea', (45.1047, 24.3750))
pitesti = Cidade('Pitesti', (44.8563, 24.8696))
bucharest = Cidade('Bucharest', (44.4323, 26.1063))
sibiu = Cidade('Sibiu', (45.8035, 24.1450))
fagaras = Cidade('Fagaras', (45.8416, 24.9731))
oradea = Cidade('Oradea', (47.0485, 21.9189))

#Criar suas conexões
graph = nx.Graph()
graph.add_edge(arad, zerind, weight=75)
graph.add_edge(arad, timisoara, weight=118)
graph.add_edge(arad, sibiu, weigth=140)
graph.add_edge(zerind, oradea, weigth=71)
graph.add_edge(timisoara, lugoj, weigth=111)
graph.add_edge(lugoj, mehadia, weigth=70)
graph.add_edge(mehadia, dobreta, weigth=75)
graph.add_edge(dobreta, craiova, weigth=120)
graph.add_edge(craiova, rimnicu_vilcea, weigth=146)
graph.add_edge(rimnicu_vilcea, sibiu, weigth=80)
graph.add_edge(rimnicu_vilcea, pitesti, weigth=97)
graph.add_edge(sibiu, fagaras, weigth=99)
graph.add_edge(fagaras, bucharest, weigth=211)
graph.add_edge(pitesti, bucharest, weigth=101)
graph.add_edge(craiova, pitesti, weigth=138)
