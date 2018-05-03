import csv
from collections import defaultdict
import math
import numpy
from copy import copy

iteracoes_maximas = 1000
convergencia = 0.01
total_k = 3

def inicializa_centroides(dados, total_k):
	centroides = defaultdict(dict)
	centroides[0] = {0: '2', 1: '1', 2: '2', 3: '1', 4: '1'}
	centroides[1] = {0: '3', 1: '1', 2: '0', 3: '0', 4: '0'}
	centroides[2] = {0: '2', 1: '0', 2: '1', 3: '2', 4: '1'}

	return centroides


def distancia_euclidiana(centroide, dado):
	total = 0

	valores_centroides = centroide
	valores_dado = dado

	for indice, c in valores_centroides.items():
		total += (int(c) - int(valores_dado[indice]))**2

	total = total**0.5
	total = round(total, 2)
	
	return total


with open('textos.csv') as arquivo:
	leitor = csv.reader(arquivo, delimiter=',')
	next(leitor)
	
	dados = defaultdict(dict)
	for i, row in enumerate(leitor):
		for j, value in enumerate(row):
			dados[i][j] = value

centroides = inicializa_centroides(dados, total_k)

for m in range(0, 10):
	matriz_distancias = defaultdict(dict)
	for i, centroide in centroides.items():
		for j, dado in dados.items():
			#print(j, dado)
			matriz_distancias[i][j] = distancia_euclidiana(centroide, dado)

	grupos = defaultdict(dict)
	for i, texto in matriz_distancias[0].items():
		aux = list()
		for j, centroide in matriz_distancias.items():
			aux.append(centroide[i])
		grupos[i] = aux.index(min(aux))

	print("%dª interacao\n" % (m))
	#print(grupos)

	for i, centroide in centroides.items():
		indices_media = list()
		for j, grupo in grupos.items():
			if grupo == i:
				#guarda os indices das posicoes que serao usadas pro calculo da media
				indices_media.append(j)

		for k, dimensao_centroide in centroide.items():
			aux = list()
			for indice_media in indices_media:
				valor = dados[indice_media][k]
				aux.append(int(valor))

			media = round(numpy.average(aux), 2)
			centroide[k] = media

	print(grupos)