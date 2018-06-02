# -*- encoding: utf-8 -*-
from sklearn import decomposition
import numpy as np

class PosProcessamento():
	def plota_dados(self, dados, centoides, grupos):
		entradas_np_array = list()
		for dado in dados:
			aux = list()
			for valor in dado:
				aux.append(valor)
			entradas_np_array.append(aux)

		dados_para_plot = np.array(list(entradas_np_array))

		centroides_np_array = list()
		for centroide in centroides:
			aux = list()
			for c in centroide:
				aux.append(c)
			centroides_np_array.append(aux)

		centroides_para_plot = np.array(list(centroides_np_array))

		grupos_para_plot = list()
		for valor in grupos.values():
			grupos_para_plot.append(valor)

		grupos_para_plot = np.array(list(grupos_para_plot))

		import matplotlib
		matplotlib.use('Agg')

		from matplotlib import pyplot as plt

		plt.rcParams['figure.figsize'] = (16, 9)
		plt.style.use('ggplot')

		pca = decomposition.PCA(n_components=2)
		pca.fit(dados_para_plot)
		dados_para_plot = pca.transform(dados_para_plot)

		pca.fit(centroides_para_plot)
		centroides_para_plot = pca.transform(centroides_para_plot)

		f1 = dados_para_plot[:, 0]
		f2 = dados_para_plot[:, 1]

		c1 = centroides_para_plot[:, 0]
		c2 = centroides_para_plot[:, 1]

		plt.scatter(c1, c2, c='red', s=14, marker='x')
		plt.scatter(f1, f2, c='black', s=7)

		plt.savefig('plot.png')
