# -*- encoding: utf-8 -*-
from means import kmeans
from means import pre_processamento
import sys
from datetime import datetime
import logging

#coleta dados dos argumentos da linha de comando
metodo = sys.argv[1]
numero_k = sys.argv[2]
calculo_distancia = sys.argv[3]
representacao = sys.argv[4]

nome_log = 'logs/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=nome_log,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

logging.info(
	'Iniciando execucao do %s, com %s clusters, usando distancia %s e dados em representacao %s.'
	% (metodo, numero_k, calculo_distancia, representacao)
)

#pre processamento de textos
pp = pre_processamento.PreProcessamento(logging)
tokens = pp.carrega_textos()
dicionario = pp.gera_dicionario(tokens)

#usa o argumento de representacao pra transformar os dados em matriz
representacoes_possiveis = ['binaria', 'tf', 'tf_idf']
if representacao not in representacoes_possiveis:
	print('A representacao precisa assumir um dos seguintes valores: [binaria, tf, tf_idf]')

dados = eval('pp.representacao_%s(dicionario, tokens)' % representacao)
logging.info('Quantidade de palavras antes da remocao das irrelevantes: %d.' % len(dados[0]))

#remove as palavras que sao irrelevantes para o clustering, reduzindo bem o numero de dimensoes
dados = pp.remove_palavras_irrelevantes(dados)
logging.info('Quantidade de palavras depois da remocao das irrelevantes: %d.' % len(dados[0]))

km = kmeans.Kmeans(logging)

#parte dos parametros recebidos para iniciar os algoritmos corretos
if metodo == 'kmeans':
	inicializacao = sys.argv[5]
	logging.info('Inicializando centroides %s.' % inicializacao)
	inicializacoes_possiveis = ['aleatoriamente', 'sobre_dados']

	if inicializacao not in inicializacoes_possiveis:
		print('A inicializacao precisa assumir um dos seguintes valores: [aleatoriamente, sobre_dados]')

	centroides = eval('km.inicializa_centroides_%s(dados, numero_k)' % inicializacao)

	grupos, centroides_finais = km.k_means(dados.copy(), centroides, numero_k)

	#cria um dict para facilitar a visualizacao dos dados por grupo
	dados_por_grupo = []
	for grupo in range(int(numero_k)):
		dados_grupo = []
		for i, dado in grupos.items():
			if dado == grupo:
				dados_grupo.append(i)
		dados_por_grupo.append(dados_grupo)

elif metodo == 'kmeans++':
	centroides = km.inicializa_k_means_mais_mais(dados.copy(), numero_k)
	grupos, centroides_finais = km.k_means(dados.copy(), centroides, numero_k)

	dados_por_grupo = []
	for grupo in range(int(numero_k)):
		dados_grupo = []
		for i, dado in grupos.items():
			if dado == grupo:
				dados_grupo.append(i)
		dados_por_grupo.append(dados_grupo)

elif metodo == 'xmeans':
	grupos, dados_por_grupo, centroides_finais = km.x_means(dados.copy(), numero_k)

else:
	print('O método precisa ter um dos seguintes valores: [kmeans, kmeans++, xmeans]')

logging.info('Centroides finais:')
logging.info(centroides_finais)
logging.info('Grupos:')
logging.info(grupos)
logging.info('Dados por grupo:')
logging.info(dados_por_grupo)

logging.info('Calculando índice silhouette:')
silhouette = km.indice_silhouette(dados.copy(), grupos)
logging.info(silhouette)

logging.getLogger('clustering')
