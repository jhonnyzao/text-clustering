import csv
from collections import defaultdict
import math

iteracoes_maximas = 1000
convergencia = 0.01
total_k = 3

def inicializa_centroides(dados, total_k):
	centroides = defaultdict(dict)
	centroides[0] = dados[1]
	centroides[1] = dados[4]
	centroides[2] = dados[6]

	return centroides

def distancia_euclidiana(centroide, dado):
	total = 0
	for index, cent in enumerate(centroide):
		total += (int(cent[i]) - int(dado[][i]))**2

	total = total^0.5
	return total

with open('textos.csv') as arquivo:
	leitor = csv.reader(arquivo, delimiter=',')
	next(leitor)
	
	dados = defaultdict(dict)
	for i, row in enumerate(leitor):
		for j, value in enumerate(row):
			dados[i][j] = value

centroides = inicializa_centroides(dados, total_k)

resultado = defaultdict(dict)
for centroide in centroides.items():
	for dado in dados.items():
		resultado[i][j] = distancia_euclidiana(centroide, dado)

print(resultado)