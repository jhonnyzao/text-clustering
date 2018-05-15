import csv
import math
import numpy
import pandas

iteracoes_maximas = 1000
convergencia = 0.01
k = 3

#with open('dados.csv') as arquivo:
#    leitor = csv.reader(arquivo, delimiter=',')
#    for row in leitor:
#        print(row[0], row[1])

#dado_a = numpy.array((xa, ya, za))
#dado_b = numpy.array((xb, yb, zb))


def aproxima(dados):
    centroides = {}
    for i in range(k):
        centroides[i] = dados[i]

    for i in range(iteracoes_maximas):
        classes = {}
        for i in range(k):
            classes[i] = []

        for dado in dados:
            distancias = [numpy.linalg.norm(dados - centroides[centroide]) for centroide in centroides]
            classificacao = distancias.index(min(distancias))
            classes[classificacao].append(dados)

        anterior = dict(centroides)

        for classificacao in classes:
            centroides[classificacao] = numpy.average(classes[classificacao], axis=0)

        convergeu = True

        for centroide in centroides:
            centroide_original = anterior[centroide]
            atual = centroides[centroide]

            if numpy.sum((atual - centroide_original) / centroide_original * 100.0) > convergencia:
                convergeu = False

        if convergeu:
            break


def pred(dados):
    distancias = [numpy.linalg.norm(dados - centroides[centroide]) for centroide in centroides]
    classificacao = distancias.index(min(distancias))

    return classificacao


#dados = pandas.read_csv(r"dados.csv")
#dados = dados[['one', 'two']]
#dataset = dados.astype(float).values.tolist()

#x = df.values

#resultado = aproxima(x)