# -*- encoding: utf-8 -*-
import csv
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import *
from collections import defaultdict
import numpy as np

class PreProcessamento:

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


	def representacao_term_frequency(self, dicionario, tokens):
		tamanho = (len(tokens), len(dicionario))
		matriz = np.zeros(tamanho)

		for i, texto in enumerate(tokens):
			for j, palavra in enumerate(texto):
				if palavra in dicionario:
					matriz[i][dicionario.index(palavra)] += 1

		return matriz


	def carrega_textos(self):
		textos = {}
		texto_index = 10
		texto_para_processar = True

		while texto_para_processar:
			try:
				nome_texto = "./textos/0%s.txt" % str(texto_index+1)
				with open(nome_texto, 'r') as arquivo:
					textos[texto_index] = arquivo.read().replace('\n', '')
					texto_index += 1
			except:
				texto_para_processar = False

		tokens = []
		for i, texto in textos.items():
			tokenizer = RegexpTokenizer(r'\w+')
			tokens.append(tokenizer.tokenize(texto))

		stop_words = set(stopwords.words("english"))
		for i, token in enumerate(tokens):
			tokens[i] = [word.lower() for word in token if word not in stop_words]

		return tokens


	def remove_palavras_irrelevantes(self, dados):
		limite_maximo = len(dados)*95/100
		limite_minimo = len(dados)*5/100

		nova_matriz = np.array()
		palavras_a_remover = list()

		for i, palavra in enumerate(dados[0]):
			contador = 0
			for j, texto in enumerate(dados):
				if palavra[j] > 0 :
					contador += 1

			if contador > limite_maximo or contador < limite_minimo:
				palavras_a_remover.append(palavra)

		palavras_a_remover = sorted(palavras_a_remover, reverse=True)

		for palavra_a_remover in palavras_a_remover:
			nova_matriz = np.delete(dados, np_s[palavra_a_remover], 1)

		return nova_matriz