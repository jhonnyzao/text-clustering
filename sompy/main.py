# -*- encoding: utf-8 -*-
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from time import time
import sompy
import os,sys,inspect
import logging

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from means import pre_processamento

nome_log = 'logs/som-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
logging.basicConfig(filename=nome_log,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


mapsize_x = sys.argv[1]
mapsize_y = sys.argv[2]
representacao = sys.argv[3]
corpora = sys.argv[4]

corporas_possiveis = ['bbcsports', 'newsgroup20']
if corpora not in corporas_possiveis:
	print('O corpora precisa assumir um dos seguintes valores: [bbcsports, newsgroup20]')

representacoes_possiveis = ['binaria', 'tf', 'tf_idf']
if representacao not in representacoes_possiveis:
	print('A representacao precisa assumir um dos seguintes valores: [binaria, tf, tf_idf]')

pp = pre_processamento.PreProcessamento(logging)
logging.info("Iniciando exeucao do SOM")

mapsize = [int(mapsize_x), int(mapsize_y)]

#print

#resgata de arquivo caso o corpora ja tenha sido pre processado antes
# nome_arquivo = 'textos_pre_processados/%s-%s.txt' % (corpora, representacao)
# try:
# 	carrega_texto_processado = np.loadtxt(nome_arquivo)
# except:
# 	carrega_texto_processado = []

# if len(carrega_texto_processado) > 0:
# 	dados = carrega_texto_processado
# 	logging.info(
# 		'Corpora ja pre processado antes. Numero de dimensoes: %d.' % len(dados[0])
# 	)

# else:
# 	tokens = eval('pp.carrega_textos_%s()' % corpora)
# 	dicionario = pp.gera_dicionario(tokens)

# 	#remove as palavras que sao irrelevantes para o clustering ou carrega arquivo ja processado
# 	#com o objetivo de trabalhar com um numero reduzido de dimensoes
# 	dados = eval('pp.representacao_%s(dicionario, tokens)' % representacao)
# 	logging.info('Quantidade de palavras antes da remocao das irrelevantes: %d.' % len(dados[0]))
	
# 	dados = pp.remove_palavras_irrelevantes(dados, corpora, representacao)
# 	logging.info('Quantidade de palavras depois da remocao das irrelevantes: %d.' % len(dados[0]))

dlen = 200
Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,2))
Data1.values[:,1] = (Data1.values[:,0][:,np.newaxis] + .42*np.random.rand(dlen,1))[:,0]

Data2 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+1)
Data2.values[:,1] = (-1*Data2.values[:,0][:,np.newaxis] + .62*np.random.rand(dlen,1))[:,0]

Data3 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+2)
Data3.values[:,1] = (.5*Data3.values[:,0][:,np.newaxis] + 1*np.random.rand(dlen,1))[:,0]


Data4 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+3.5)
Data4.values[:,1] = (-.1*Data4.values[:,0][:,np.newaxis] + .5*np.random.rand(dlen,1))[:,0]


Data1 = np.concatenate((Data1,Data2,Data3,Data4))

som = sompy.SOMFactory.build(Data1, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
som.train(n_job=1, verbose='info')

topographic_error = som.calculate_topographic_error()

print(topographic_error)