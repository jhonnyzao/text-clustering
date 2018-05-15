import csv
from collections import defaultdict
import math
import numpy
from copy import copy
from random import randint

iteracoes_maximas = 1000
convergencia = 0.01
total_k = 3

def inicializa_centroides_aleatoriamente(dados, total_k):
	centroides = defaultdict(dict)
	valores_maximos = list()
	valores_minimos = list()
	
	for linha in dados.values():
		valores_maximos.append(max(linha.values()))
		valores_minimos.append(min(linha.values()))

	valor_maximo = max(valores_maximos)
	valor_minimo = min(valores_minimos)
	
	for linha in range(0, total_k):
		for coluna in range(0, len(dados[0])):
			centroides[linha][coluna] = randint(int(valor_minimo), int(valor_maximo))

	c = defaultdict(dict)
	
	c[0] = {0: '2', 1: '1', 2: '2', 3: '1', 4: '1'}
	c[1] = {0: '3', 1: '1', 2: '0', 3: '0', 4: '0'}
	c[2] = {0: '2', 1: '0', 2: '1', 3: '2', 4: '1'}

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

centroides = inicializa_centroides_aleatoriamente(dados.copy(), total_k)

grupos = defaultdict(dict)
grupos_ultima_iteracao = defaultdict(dict)
convergiu = False
iteracao_atual = 0

while (iteracao_atual <= iteracoes_maximas or not convergiu):
	matriz_distancias = defaultdict(dict)
	for i, centroide in centroides.items():
		for j, dado in dados.items():
			matriz_distancias[i][j] = distancia_euclidiana(centroide, dado)

	grupos_ultima_iteracao = grupos.copy()

	for i, texto in matriz_distancias[0].items():
		aux = list()
		for j, centroide in matriz_distancias.items():
			aux.append(centroide[i])
		grupos[i] = aux.index(min(aux))

	#compara se algum dado mudou de centroide e assume convergencia em caso negativo
	if grupos_ultima_iteracao == grupos:
		convergiu = True
		break

	print("%dª iteracao\n" % (iteracao_atual))
	
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

			if aux:
				media = round(numpy.average(aux), 2)
				centroide[k] = media
	print(grupos)
	iteracao_atual += 1