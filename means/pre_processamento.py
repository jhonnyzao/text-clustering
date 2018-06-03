# -*- encoding: utf-8 -*-
import os
import csv
from collections import Counter
from collections import defaultdict
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import *
import itertools, collections

class PreProcessamento:
    def __init__(self, logging):
        self.logging = logging


    def gera_dicionario(self, tokens):
        dicionario = []

        for doc in tokens:
            for palavra in doc:
                if palavra not in dicionario:
                    dicionario.append(palavra)
        
        return dicionario


    def representacao_binaria(self, dicionario, tokens):
        tamanho = (len(tokens), len(dicionario))
        matriz = np.zeros(tamanho)

        for i, texto in enumerate(tokens):
            for j, palavra in enumerate(texto):
                if palavra in dicionario:
                    matriz[i][dicionario.index(palavra)] = 1

        return matriz


    def representacao_tf(self, dicionario, tokens):
        tamanho = (len(tokens), len(dicionario))
        matriz = np.zeros(tamanho)

        for i, texto in enumerate(tokens):
            for j, palavra in enumerate(texto):
                if palavra in dicionario:
                    matriz[i][dicionario.index(palavra)] += 1

        return matriz


    def idf(self, palavra, corpus):
        cont = 0
        qtd_docs = len(corpus)
        for doc in corpus:
            if doc.split().count(palavra) > 0:
                cont += 1
        return np.log(qtd_docs/cont)


    def representacao_tf_idf(self, dicionario, tokens, corpus):
        tamanho = (len(tokens), len(dicionario))
        matriz = np.zeros(tamanho)
        matriz_tf = representacao_term_frequency(dicionario, tokens)

        for i, texto in enumerate(tokens):
            for j, palavra in enumerate(texto):
                if palavra in dicionario:
                    matriz = matriz_tf[i][dicionario.index(palavra)] * idf(palavra, corpus)

        return matriz


    def carrega_textos_newsgroup20(self):
        textos = {}
        texto_index = 0

        def consume(iterator, n): collections.deque(itertools.islice(iterator, n))

        with open('big_corpora/newgroup20.txt', 'r') as arquivo:
            conteudo = arquivo.readlines()

        texto = ''
        iterator = conteudo.__iter__()

        #as linhas que comecam com a palavra 'newsgroup:' separam os textos, seguidas de mais
        #3 linhas de cabecalho. transformar todo o conteudo lido em um iterator ajuda nesse
        #processo
        for linha in iterator:
            if 'Newsgroup:' in linha:
                consume(iterator, 3)
                if texto:
                    texto = texto.replace('\n', '')
                    textos[texto_index] = texto
                    texto_index += 1
                texto = ''
            else:
                texto += linha

        self.logging.info("Iniciando com dataset de %d dados." % len(textos))
        tokens = []
        #faz o processo de tokenizacao da nltk
        for i, texto in textos.items():
            tokenizer = RegexpTokenizer(r'\w+')
            tokens.append(tokenizer.tokenize(texto))

        #usa a lista de stopwords pronta da nlkt para o ingles e remove os tokens que as contem
        stop_words = set(stopwords.words("english"))
        for i, token in enumerate(tokens):
            tokens[i] = [word.lower() for word in token if word not in stop_words]

        return tokens


    def carrega_textos_bbcsports(self):
        textos = {}
        texto_index = 0

        for root, dirs, files in os.walk(r'C:\\Users\\jhonnyzao\\Documents\\clustering\\textos\\bbcsport\\'):
        #for root, dirs, files in os.walk('/home/joao/text-clustering/textos/bbcsport/'):
            for file in files:
                with open(os.path.join(root, file), "rb") as arquivo:
                    texto = str(arquivo.read())
                    texto = texto.replace('\\n', '')
                    textos[texto_index] = texto
                    texto_index += 1

        self.logging.info("Iniciando com dataset de %d dados." % len(textos))
        tokens = []
        #faz o processo de tokenizacao da nltk
        for i, texto in textos.items():
            tokenizer = RegexpTokenizer(r'\w+')
            tokens.append(tokenizer.tokenize(texto))

        #usa a lista de stopwords pronta da nlkt para o ingles e remove os tokens que as contem
        self.logging.info("Removendo stop words")
        stop_words = set(stopwords.words("english"))
        for i, token in enumerate(tokens):
            tokens[i] = [word.lower() for word in token if word not in stop_words]

        return tokens


    def remove_palavras_irrelevantes(self, dados, corpora, representacao):
        self.logging.info('Removendo palavras irrelevantes')
        #limites representam a porcentagem de textos em que cada palavra aparece
        limite_maximo = len(dados[0])*99/100
        limite_minimo = len(dados[0])*1/100

        nova_matriz = dados
        palavras_a_remover = list()

        #percorre todas as colunas da matriz de dados (cada coluna representa uma palavra)
        for i in range(len(dados[0])):
            contador = 0
            #percorre cada um dos textos para cada palavra e incrementa a variavel de controle caso ela
            #esteja presente no texto
            for j, texto in enumerate(dados):
                if texto[i] > 0:
                    contador += 1

            #se a palavra no estiver dentro dos limites minimos e maximos, ela nao eh
            #representativa para a massa e nao deve ser utilizada para fins de clustering
            if contador >= limite_maximo or contador <= limite_minimo:
                palavras_a_remover.append(i)

        #ordena as palavras a serem removidas de forma decrescente para o removedor nao
        #se perder com os indices
        palavras_a_remover = sorted(palavras_a_remover, reverse=True)

        #remove cada uma das palavras irrelevantes pro clustering
        for palavra_a_remover in palavras_a_remover:
            nova_matriz = np.delete(nova_matriz, np.s_[palavra_a_remover], 1)

        nome_arquivo = 'textos_pre_processados/%s-%s.txt' % (corpora, representacao)
        np.savetxt(nome_arquivo, nova_matriz)

        return nova_matriz
