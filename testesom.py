# -*- encoding: utf-8 -*-
from sompy.sompy import sompy
from pre_processamento import *

pp = PreProcessamento()
tokens = pp.carrega_textos()
dicionario = pp.gera_dicionario(tokens)
dados = pp.representacao_term_frequency(dicionario, tokens)
dados = pp.remove_palavras_irrelevantes(dados)

print(dados)

mapsize = [40,40]
som = sompy.SOMFactory.build(dados, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
som.train(n_job=1, verbose='info')
