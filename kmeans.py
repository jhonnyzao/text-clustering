# -*- encoding: utf-8 -*-
import csv
from collections import defaultdict
import math
import numpy as np
from copy import copy
from random import randint
import random
from pre_processamento import *
from sklearn import decomposition

def inicializa_centroides_aleatoriamente(dados, total_k):
	dimensao = (total_k, len(dados[0]))
	centroides = np.zeros(dimensao)

	valores_maximos = list()
	valores_minimos = list()

	for linha in dados:
		valores_maximos.append(max(linha))
		valores_minimos.append(min(linha))

	valor_maximo = max(valores_maximos)
	valor_minimo = min(valores_minimos)

	for linha in range(0, total_k):
		for coluna in range(0, len(dados[0])):
			centroides[linha][coluna] = randint(int(valor_minimo), int(valor_maximo))

	return centroides

def inicializa_centroides_sobre_dados(dados, total_k):
	#escolhe k dados distintos para inicializacao dos centroides sobre eles
	dados_escolhidos = random.sample(range(1, len(dados)), total_k)

	centroides = []
	#popula cada centroide com uma copia dos dados sorteados
	for i, posicao_dado in enumerate(dados_escolhidos):
		centroides.append(dados[posicao_dado].copy())

	return centroides


def inicializa_k_means_mais_mais(dados_copia, total_k):
	centroides = defaultdict(dict)

	if total_k < 1:
		print('O número de centroides escolhidos precisa ser maior do que 0')

	#posiciona o primeiro centroide sobre um dado aleatorio
	centroides[0] = dados_copia[randint(0, int(len(dados))-1)]

	#partindo entao do segundo centroide ate o ultimo
	for k in range(1, total_k):
		distancias = defaultdict(dict)
		#monta uma matriz de distancias onde cada linha representa um dado
		#com sua respectiva distancia do mais proximo dos centroides ja escolhidos
		for x, dado in enumerate(dados_copia):
			distancias_centroides_escolhidos = list()
			for y, centroide in centroides.items():
				distancias_centroides_escolhidos.append(round(distancia_euclidiana(centroide, dado), 2))
			minimo = min(distancias_centroides_escolhidos)
			#o algoritmo assume distancias elevadas ao quadrado
			distancias[x] = round(minimo ** 2, 2)

		#a soma das distancias eh usada para o calculo das probabilidades
		soma_distancias = round(sum(distancias.values()), 2)

		#monta uma matriz de probabilidades, sendo elas calculadas pela distancia do dado pro centroide dividido pela soma das distancias
		probabilidades = defaultdict(dict)
		for i, distancia in distancias.items():
			probabilidades[i] = round(distancia/soma_distancias, 2)

		aux = list()
		for i, probabilidade in probabilidades.items():
			aux.append(probabilidade)

		#calcula o cumulativo de cada probabilidade com as probabilidades anteriores
		probabilidades_cumulativas = np.cumsum(aux)

		#define valor aleatorio para comparar com a probabilidade
		#o valor vai de 0 a 1.01 pois podem existir probabilidades cumulativas no valor de 1.01 devido aos arrendodamentos
		aleatorio =  round(random.uniform(0,1.01), 2)

		#percorre cada valor do array de probabilidades acumuladas
		for j, probabilidade in enumerate(probabilidades_cumulativas):
			#o if abaixo compara o cumulativo de probabilidades com um numero aleatorio, a fim de tentar encontrar o dado
			#mais longe dos centroides ja alocados
			if aleatorio < probabilidade:
				#o dado na posicao j pode ser o dado mais longe dos centroides ja alocados ou nao, mas o algoritmo
				# garante que ele nao esta perto o suficiente para ser considerado uma boa posicao para inicializar um centroide
				centroides[k] = dados_copia[j]
				break

	return centroides


def indice_silhouette(dados, grupos):
	silhouettes = defaultdict(dict)

	valores_mesmo_cluster = list()
	valores_cluster_diferente = list()

	for i, dado in enumerate(dados):
		#variavel que guarda dado ja existente em outra variavel, mas facilita a leitura do codigo
		grupo_atual = grupos[i]
		#para cada dado, percorre toda a matriz que associa dado a centroide, e guarda as distancias euclidianas
		#nas listas de valores do mesmo cluster ou de cluster diferente de acordo com o cenario
		for j, grupo in grupos.items():
			if grupo == grupo_atual:
				distancia = distancia_euclidiana(dado, dados[j])
				if distancia != 0:
					valores_mesmo_cluster.append(distancia)
			else:
				valores_cluster_diferente.append(distancia_euclidiana(dado, dados[j]))

	#na literatura, b(i) eh o nome da variavel que guarda a distancia media dos dados em um
	#centroide para todos os outros dados de centroides diferentes	
	b = round(sum(valores_cluster_diferente)/len(valores_cluster_diferente), 2)

	#na literatura, a(i) eh o nome da variavel que guarda a distancia media dos dados em um
	#centroide para todos os demais dados no mesmo centroide
	a = round(sum(valores_mesmo_cluster)/len(valores_mesmo_cluster), 2)

	indice_silhouette = round((b - a)/max(a, b), 2)
	print(indice_silhouette)
	

def distancia_euclidiana(centroide, dado):
	total = 0

	for i, valor_centroide in enumerate(centroide):
		total += (valor_centroide - dado[i])**2

	total = total**0.5
	total = round(total, 2)

	return total


def obtem_matriz_distancias(centroides, dados):
	matriz_distancias = defaultdict(dict)
	#dois loops aninhados para comparar a distancia de cada centroide com cada dado
	for i, centroide in enumerate(centroides):
		for j, dado in enumerate(dados):
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
	for i, centroide in enumerate(centroides):
		indices_media = list()
		#percorre o dict de grupos procurando todos os dados associados ao centroide i, que sao importantes para o calculo de sua nova posicao
		for j, grupo in grupos.items():
			if grupo == i:
				#guarda os indices das posicoes que serao usadas pro calculo da media
				indices_media.append(j)

		for k, dimensao_centroide in enumerate(centroide):
			aux = list()
			#percorre a matriz de dados pegando a posicao de todos os que estao na lista indices_media, ou seja, todos os dados associados ao centroide i
			for indice_media in indices_media:
				valor = dados[indice_media][k]
				aux.append(int(valor))

			#verificacao necessaria para nao quebrar o programa caso nao haja nenhum dado no grupo de algum centroide
			if aux:
				media = round(np.average(aux), 2)
				centroide[k] = media

pp = PreProcessamento()

tokens = pp.carrega_textos()
dicionario = pp.gera_dicionario(tokens)
dados = pp.representacao_binaria(dicionario, tokens)

iteracoes_maximas = 1000
total_k = 20

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

	iteracao_atual += 1


indice_silhouette(dados, grupos)
# import matplotlib
# from mpl_toolkits.mplot3d import Axes3D

#matplotlib.use('Agg')

#from matplotlib import pyplot as plt

entradas_np_array = list()
for dado in dados:
    aux = list()
    for valor in dado:
        aux.append(valor)
    entradas_np_array.append(aux)

dados_para_plot = np.array(list(entradas_np_array))

grupos_para_plot = list()
for valor in grupos.values():
    grupos_para_plot.append(valor)

grupos_para_plot = np.array(list(grupos_para_plot))

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# fig = plt.figure(1, figsize = (4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(dados_para_plot)
dados_para_plot = pca.transform(dados_para_plot)


f1 = dados_para_plot[:, 0]
f2 = dados_para_plot[:, 1]
plt.scatter(f1, f2, c='black', s=7)
# for name, label in [('Um', 0), ('Dois', 1), ('Tres', 2)]:
#     ax.text3D(dados_para_plot[grupos_para_plot == label, 0].mean(),
#               dados_para_plot[grupos_para_plot == label, 1].mean() + 1.5,
#               dados_para_plot[grupos_para_plot == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# grupos_para_plot = np.choose(grupos_para_plot, [1, 2, 0]).astype(np.float)
# ax.scatter(dados_para_plot[:, 0], dados_para_plot[:, 1], dados_para_plot[:, 2], c=grupos_para_plot, edgecolor='k')

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

plt.savefig('plot.png')
