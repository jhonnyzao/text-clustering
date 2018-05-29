# -*- encoding: utf-8 -*-
import csv
from collections import defaultdict
import math
import numpy as np
from copy import copy
from random import randint
import random
from pre_processamento import *

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
	centroides = np.zeros((total_k, len(dados[0])))

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
			for y, centroide in enumerate(centroides):
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
	grupos = defaultdict(dict)

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
				aux.append(valor)

			#verificacao necessaria para nao quebrar o programa caso nao haja nenhum dado no grupo de algum centroide
			if aux:
				media = round(np.average(aux), 2)
				centroide[k] = media


def k_means(dados, centroides, total_k):
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

	return grupos, centroides


def x_means(dados):
	k_inicial = 2
	centroides_iniciais = inicializa_k_means_mais_mais(dados.copy(), k_inicial)
	grupos, c = k_means(dados, centroides_iniciais, k_inicial)

	dados_por_grupo = defaultdict(dict)

	#percorre o dict de grupos e monta um novo dict para facilitar a manipulacao dos dados
	#nessa nova estrutura, um dict de k linhas representam k centroides, e seus respectivos
	#conteudos sao os indices dos dados a eles associados
	for grupo in range(k_inicial):
		dados_grupo = []
		for i, dado in grupos.items():
			if dado == grupo:
				dados_grupo.append(i)
		dados_por_grupo[grupo] = dados_grupo

	centroides = []
	for centroide_inicial in centroides_iniciais:
		centroides.append((False, centroide_inicial))

	centroides_estado_final = False
	while (not centroides_estado_final):
		for i, centroide in enumerate(centroides):
			if not centroide[0]:
				bic_centroide_pai = calcula_bic(dados, [dados_por_grupo[i]], [centroide[1]])
				dados_centroide_pai = [[dados[dado] for dado in dado_por_grupo] for dado_por_grupo in [dados_por_grupo[i]]]

				#quebra o centroide atual em dois
				novos_centroides = fragmenta_centroide_em_dois(dados_centroide_pai[0], centroide[1])
				#passa o kmeans localmente nos dois novos centroides
				novos_grupos, novos_centroides = k_means(dados_centroide_pai[0].copy(), novos_centroides, 2)
				novos_dados_por_grupo = []
				for j in range(2):
					novo_dado_por_grupo = []
					for k, dado in novos_grupos.items():
						if dado == j:
							novo_dado_por_grupo.append(k)
					novos_dados_por_grupo.append(novo_dado_por_grupo)

				bic_filhos = calcula_bic(dados, novos_dados_por_grupo, novos_centroides)
				print(bic_centroide_pai)
				print(bic_filhos)
				exit()

				bic_c1 = calcula_bic(carga_dados_do_grupo, novos_centroides[0][1])
				bic_c2 = calcula_bic(carga_dados_do_grupo, novos_centroides[1][1])

				if bic_c1 > bic_centroide_pai or bic_c2 > bic_centroide_pai:
					centroides.append(novos_centroides[0])
					centroides.append(novos_centroides[1])
				else:
					centroide[0] = True

		if not False in [c[0] for c in centroides]:
			centroides_estado_final = True


def calcula_bic(dados, dados_por_grupo, centroides):
	print(dados)
	print(dados_por_grupo)
	print(centroides)
	carga_dados_do_grupo = [[dados[dado] for dado in dado_por_grupo] for dado_por_grupo in dados_por_grupo]

	variancia = calcula_variancia_clusters(dados, carga_dados_do_grupo, centroides)
	constante = 0.5 * len(dados_por_grupo) * np.log(len(dados)) * len(dados[0]+1)

	bic = 0
	#formula de calculo retirada da combinacao da literatura com uma discussao em
	#foruns de IA, ambos referenciados no relatorio
	for i in range(len(centroides)):
		bic += (len(dados_por_grupo[i]) * np.log(len(dados_por_grupo[i]))) - \
		(len(dados_por_grupo[i]) * np.log(len(dados))) - \
		((len(dados_por_grupo[i]) * len(dados[0])) / 2) * \
		np.log(2 * np.pi * variancia) - \
		((len(dados_por_grupo[i]) - 1) * \
		len(dados[0]) / 2)

	bic = bic - constante

	return bic


def calcula_variancia_clusters(dados, dados_por_grupo, centroides):
	#no calculo da variacia, o denominador é formado pela quantidade de dados menos a quantidade de
	#grupos multiplicado pela quantidade de dimensoes
	denominador = (len(dados) - len(dados_por_grupo)) * len(dados[0])

	soma_todas_distancias = 0

	#incrementa o quadrado da distancia euclidiana de todos os dados para todos os centroides
	for i, dados_grupo in enumerate(dados_por_grupo):
		soma_distancias_grupo = 0
		for dado in dados_grupo:
			soma_distancias_grupo += distancia_euclidiana(centroides[i], dado)**2

		soma_todas_distancias += soma_distancias_grupo

	#final da formula da variania, que eh a soma total das distancias quadradas sobre o denominador
	variancia = soma_todas_distancias/denominador

	return variancia


def fragmenta_centroide_em_dois(dados, centroide):
	novo_c1 = []
	novo_c2 = []

	#percorre cada atributo do centroide
	for i, dimensao in enumerate(centroide):

		#encontra o dado que tem o maior valor para o atributo dessa iteracao
		valor_maximo_dado = max(dado[i] for dado in dados)

		#o centroide antigo eh encarado com um repelente dos novos centroides, ou seja,
		#ele inicializa um mais proximo do valor maximo do grupo (o dado com maior valor pro atributo)
		#e outro no sentido oposto, criando um grande vetor ligando as duas bordas e sendo
		#dividido por 3
		distancia_a_percorrer = (valor_maximo_dado - dimensao)/3

		novo_c1.append(round(dimensao + distancia_a_percorrer, 2))
		novo_c2.append(round(dimensao - distancia_a_percorrer, 2))

	novos_centroides = [novo_c1, novo_c2]

	return novos_centroides


iteracoes_maximas = 1000
total_k = 8

pp = PreProcessamento()
tokens = pp.carrega_textos()
dicionario = pp.gera_dicionario(tokens)
dados = pp.representacao_binaria(dicionario, tokens)
dados = pp.remove_palavras_irrelevantes(dados)

#eh importante passar uma copia do dict de dados para que a matriz de dados original nao seja
#alterada durante as movimentacoes dos centroides
#centroides = inicializa_k_means_mais_mais(dados.copy(), total_k)

grupos = x_means(dados)

indice_silhouette(dados, grupos)
