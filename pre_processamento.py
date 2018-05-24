# -*- encoding: utf-8 -*-
import csv
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import *
from collections import defaultdict

textos = {}
stop_words = set(stopwords.words("english"))

def gera_matriz_binaria(tokens):
	matriz = defaultdict(dict)
	labels = defaultdict(dict)
	index = 0

	for doc in tokens:
		for palavra in doc:
			if palavra not in labels:
				ultima_posicao = len(labels)
				labels[ultima_posicao] = palavra
				matriz[index][ultima_posicao] = 1

		index += 1

	return matriz

texto_index = 0
texto_para_processar = True

while texto_para_processar:
	try:
		nome_texto = "./textos/00%s.txt" % str(texto_index+1)
		with open(nome_texto, 'r') as arquivo:
			textos[texto_index] = arquivo.read().replace('\n', '')
			texto_index += 1
	except:
		texto_para_processar = False

tokens = []
for i, texto in textos.items():
	tokenizer = RegexpTokenizer(r'\w+')
	tokens.append(tokenizer.tokenize(texto))

for i, token in enumerate(tokens):
	tokens[i] = [word.lower() for word in token if word not in stop_words]

matriz_binaria = gera_matriz_binaria(tokens)

for m in matriz_binaria.values():
	print(m) 
# with open('dados_finais.csv', 'wb') as csvfile:
# 	spamwriter = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
#     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])