import csv
from collections import defaultdict
import math
import numpy

iteracoes_maximas = 1000
convergencia = 0.01
total_k = 3

def inicializa_centroides(dados, total_k):
	centroides = defaultdict(dict)
	centroides[0] = dados[1]
	centroides[1] = dados[4]
	centroides[2] = dados[6]

	return centroides


def distancia_euclidiana(centroide, dado):
	total = 0

	valores_centroides = centroide
	valores_dado = dado

	for indice, c in valores_centroides.items():
		print(c, valores_dado[indice])
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
			matriz_distancias[i][j] = distancia_euclidiana(centroide, dado)

	grupos = defaultdict(dict)
	for i, texto in matriz_distancias[0].items():
		aux = list()
		for j, centroide in matriz_distancias.items():
			aux.append(centroide[i])
		grupos[i] = aux.index(min(aux))

	print("%dª interacao\n" % (m))
	print(grupos)

	for i, centroide in centroides.items():
		indices_media = list()
		for j, grupo in grupos.items():
			if grupo == i:
				#guarda os indices das posicoes que serao usadas pro calculo da media
				indices_media.append(j)
		aux = list()
		for k, dimensao_centroide in centroide.items():
			for indice_media in indices_media:
				valor = round(float(dados[indice_media][k]), 2)
				aux.append(valor)

			media = round(numpy.average(aux), 2)
			centroide[k] = media
