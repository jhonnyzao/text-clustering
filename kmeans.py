# -*- encoding: utf-8 -*-
import csv
from collections import defaultdict
import math
import numpy
from copy import copy
from random import randint
import random

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

	return centroides

def inicializa_centroides_sobre_dados(dados, total_k):
	#escolhe k dados distintos para inicializacao dos centroides sobre eles
	dados_escolhidos = random.sample(range(1, len(dados)), total_k)

	centroides = defaultdict(dict)
	#popula cada centroide com uma copia dos dados sorteados
	for i, posicao_dado in enumerate(dados_escolhidos):
		centroides[i] = dados[posicao_dado].copy()

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


def obtem_matriz_distancias(centroides, dados):
	matriz_distancias = defaultdict(dict)
	#dois loops aninhados para comparar a distancia de cada centroide com cada dado
	for i, centroide in centroides.items():
		for j, dado in dados.items():
			matriz_distancias[i][j] = distancia_euclidiana(centroide, dado)

	return matriz_distancias


def obtem_formacao_dos_grupos(matriz_distancias):
	#percorre cada coluna
	for i, texto in matriz_distancias[0].items():
		aux = list()
		for j, centroide in matriz_distancias.items():
			aux.append(centroide[i])
		grupos[i] = aux.index(min(aux))

	return grupos


def reposiciona_centroides(centroides, grupos, dados):
	for i, centroide in centroides.items():
		indices_media = list()
		#percorre o dict de grupos procurando todos os dados associados ao centroide i, que sao importantes para o calculo de sua nova posicao
		for j, grupo in grupos.items():
			if grupo == i:
				#guarda os indices das posicoes que serao usadas pro calculo da media
				indices_media.append(j)

		for k, dimensao_centroide in centroide.items():
			aux = list()
			#percorre a matriz de dados pegando a posicao de todos os que estao na lista indices_media, ou seja, todos os dados associados ao centroide i
			for indice_media in indices_media:
				valor = dados[indice_media][k]
				aux.append(int(valor))

			#verificacao necessaria para nao quebrar o programa caso nao haja nenhum dado no grupo de algum centroide
			if aux:
				media = round(numpy.average(aux), 2)
				centroide[k] = media


iteracoes_maximas = 1000
total_k = 3

#le o csv e coloca a posicao dos dados no dict dados
with open('textos.csv') as arquivo:
	leitor = csv.reader(arquivo, delimiter=',')
	next(leitor)

	dados = defaultdict(dict)
	for i, row in enumerate(leitor):
		for j, value in enumerate(row):
			dados[i][j] = value

#eh importante passar uma copia do dict de dados para que a matriz de dados original nao seja alterada durante as movimentacoes dos centroides
centroides = inicializa_centroides_sobre_dados(dados.copy(), total_k)

grupos = defaultdict(dict)
grupos_ultima_iteracao = defaultdict(dict)
convergiu = False
iteracao_atual = 0

#duas condicoes de parada
while (iteracao_atual <= iteracoes_maximas or not convergiu):
	matriz_distancias = obtem_matriz_distancias(centroides, dados)

	#variavel que guarda ultimo estado de grupos para analise de convergencia
	grupos_ultima_iteracao = grupos.copy()

	grupos = obtem_formacao_dos_grupos(matriz_distancias)

	#compara se algum dado mudou de centroide e assume convergencia em caso negativo
	if grupos_ultima_iteracao == grupos:
		convergiu = True
		break

	print("%dª iteracao\n" % (iteracao_atual))

	reposiciona_centroides(centroides, grupos, dados)

	print(grupos)
	iteracao_atual += 1
