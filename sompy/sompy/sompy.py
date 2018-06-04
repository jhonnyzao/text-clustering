# -*- coding: utf-8 -*-

# Author: Vahid Moosavi (sevamoo@gmail.com)
#         Chair For Computer Aided Architectural Design, ETH  Zurich
#         Future Cities Lab
#         www.vahidmoosavi.com

# Contributor: Sebastian Packmann (sebastian.packmann@gmail.com)


import tempfile
import os
import itertools
import logging

import numpy as np

from time import time
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from sklearn import neighbors
from sklearn.externals.joblib import Parallel, delayed, load, dump
import sys

from .decorators import timeit
from .codebook import Codebook
from .neighborhood import NeighborhoodFactory
from .normalization import NormalizatorFactory

#lbugnon
#import ipdb
import sompy
#

class ComponentNamesError(Exception):
    pass


class LabelsError(Exception):
    pass


class SOMFactory(object):

    """
    Inicializacao da SOM, recebendo seus parametros, onde:
        - data: eh a matriz de dados, de m linhas e n colunas. cada linha representa um dado,
        e cada coluna representa um atributo do dado
        - mapsize: sao as dimensoes da SOM, que usamos sempre em formato de matriz pois queremos
        uma SOM bidimensional. valores diferentes serao usados no trabalho. ex: '[10,10]'
        - mask: parametro opcional para o pos processamento, nao utilizados no trabalho
        - mapshape: define a forma do SOM. no trabalho, vamos usar sempre uma SOM plana, logo,
        com esse parametro recebe sempre o valor 'planar'
        - lattice: formato da lattice, usada durante os ajustes na vizinhanca. pode
        ser hexagonal ou retangular. vamos usar sempre 'rect'
        - normalization: tipo de normalizacao usada para os calculos de aproximacao. vamos sempre usar
        var, que significa variancia
        - initialization: define qual metodo utilizar para inicializar a SOM. pode ser
        'random' ou 'pca', e vamos usar ambos no trabalho
        - neighborhood: qual metodo sera utilizado para o calculo da vizinhanca no som. pode ser 
        'gaussian' ou 'bubble'
        - training: modo de treinamento. pode ser seq ou batch, indicando se o mesmo acontecera
        sequencialmente ou em blocos. vamos usar 'batch'
        - name: um nome que podemos atribuir a SOM
        - component_names: os nomes dos componentes, caso existam. nao vamos definir nada
        antes do pos processamento, mas no nosso caso seriam as palavras (colunas dos dados)
    """
    @staticmethod
    def build(data,
              mapsize=None,
              mask=None,
              mapshape='planar',
              lattice='rect',
              normalization='var',
              initialization='pca',
              neighborhood='gaussian',
              training='batch',
              name='sompy',
              component_names=None):

        #chama as factories do normalizador e do calculador de vizinhanca escolhidos
        #para serem usados posteriormente
        if normalization:
            normalizer = NormalizatorFactory.build(normalization)
        else:
            normalizer = None
        neighborhood_calculator = NeighborhoodFactory.build(neighborhood)

        return SOM(data, neighborhood_calculator, normalizer, mapsize, mask,
                   mapshape, lattice, initialization, training, name, component_names)


class SOM(object):

    def __init__(self,
                 data,
                 neighborhood,
                 normalizer=None,
                 mapsize=None,
                 mask=None,
                 mapshape='planar',
                 lattice='rect',
                 initialization='pca',
                 training='batch',
                 name='sompy',
                 component_names=None):

        self._data = normalizer.normalize(data) if normalizer else data
        self._normalizer = normalizer
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self._dlabel = None
        self._bmu = None

        self.name = name
        self.data_raw = data
        self.neighborhood = neighborhood
        self.mapshape = mapshape
        self.initialization = initialization
        self.mask = mask or np.ones([1, self._dim])
        mapsize = self.calculate_map_size(lattice) if not mapsize else mapsize
        self.codebook = Codebook(mapsize, lattice)
        self.training = training
        self._component_names = self.build_component_names() if component_names is None else [component_names]
        self._distance_matrix = self.calculate_map_dist()

    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        if self._dim == len(compnames):
            self._component_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ComponentNamesError('Os nomes dos componentes precisam ter '
                                      'a mesma quantidade que o numero de colunas dos dados')

    def build_component_names(self):
        cc = ['Variable-' + str(i+1) for i in range(0, self._dim)]
        return np.asarray(cc)[np.newaxis, :]

    @property
    def data_labels(self):
        return self._dlabel

    @data_labels.setter
    def data_labels(self, labels):
        """
        Define os formatos das labels. precisa ser lista ou strings
        """
        #as labels podem estar em formato 'um por linha da matriz' ou 'um por coluna'
        #ou seja, podem indicar os dados ou atributos
        if labels.shape == (1, self._dlen):
            label = labels.T
        elif labels.shape == (self._dlen, 1):
            label = labels
        elif labels.shape == (self._dlen,):
            label = labels[:, np.newaxis]
        else:
            raise LabelsError('Formato invalido de labels')

        self._dlabel = label

    def build_data_labels(self):
        cc = ['dlabel-' + str(i) for i in range(0, self._dlen)]
        return np.asarray(cc)[:, np.newaxis]

    def calculate_map_dist(self):
        """
        Calculates the grid distance, which will be used during the training
        steps. It supports only planar grids for the moment
        """
        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
        return distance_matrix


    """
    Funcao que faz o treinamento da SOM. Parametros relevantes pro trabalho:
        - n_job: numero de jobs usados para paralelizar o treinamento. o codigo, por enquanto,
        so suporta 1
        - verbose: o nivel de debug do treinamento. pode assumir 'debug' ou 'info'. vamos usar info
        para obter apenas interessantes durante a execucao
        - shared_memory: flag pra controlar a ativação do compartilhamento de memoria, que nao
        vamos usar
    """
    @timeit()
    def train(self,
              n_job=1,
              shared_memory=False,
              verbose='info',
              train_rough_len=None,
              train_rough_radiusin=None,
              train_rough_radiusfin=None,
              train_finetune_len=None,
              train_finetune_radiusin=None,
              train_finetune_radiusfin=None,
              train_len_factor=1,
              maxtrainlen=np.Inf):

        #seta nivel de debug conforme escolhido
        logging.root.setLevel(
            getattr(logging, verbose.upper()) if verbose else logging.ERROR)

        logging.info(" Training...")
        logging.debug((
            "--------------------------------------------------------------\n"
            " details: \n"
            "      > data len is {data_len} and data dimension is {data_dim}\n"
            "      > map size is {mpsz0},{mpsz1}\n"
            "      > array size in log10 scale is {array_size}\n"
            "      > number of jobs in parallel: {n_job}\n"
            " -------------------------------------------------------------\n")
            .format(data_len=self._dlen,
                    data_dim=self._dim,
                    mpsz0=self.codebook.mapsize[0],
                    mpsz1=self.codebook.mapsize[1],
                    array_size=np.log10(
                        self._dlen * self.codebook.nnodes * self._dim),
                    n_job=n_job))

        #faz a inicializacao conforme especificacao nos parametros
        if self.initialization == 'random':
            self.codebook.random_initialization(self._data)

        elif self.initialization == 'pca':
            self.codebook.pca_linear_initialization(self._data)

        #com todos os parametros definidos, chama finalmente os metodos de treinamento
        self.rough_train(njob=n_job, shared_memory=shared_memory, trainlen=train_rough_len,
                         radiusin=train_rough_radiusin, radiusfin=train_rough_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        self.finetune_train(njob=n_job, shared_memory=shared_memory, trainlen=train_finetune_len,
                            radiusin=train_finetune_radiusin, radiusfin=train_finetune_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        logging.debug(
            " --------------------------------------------------------------")
        logging.info(" Final quantization error: %f" % np.mean(self._bmu[1]))

    def _calculate_ms_and_mpd(self):
        mn = np.min(self.codebook.mapsize)
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])

        if mn == 1:
            mpd = float(self.codebook.nnodes*10)/float(self._dlen)
        else:
            #nos exemplos, sempre entramos no else, que calcula o total de neuronios da SOM
            #dividido pela quantidade de dados
            mpd = float(self.codebook.nnodes)/float(self._dlen)
        ms = max_s/2.0 if mn == 1 else max_s

        return ms, mpd

    #faz o "treinamento pesado", que eh a primeira parte do treinamento da SOM
    def rough_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Rough training...")

        #variaveis que guardam o tamanho da SOM, onde ms significa mapsize e mpd significa
        #o numero de neuronios por dado
        ms, mpd = self._calculate_ms_and_mpd()
        
        #caso nao esteja pre setada um tamanho maximo de treino, seta na hora
        #estamos trabalhando sempre com esse valor infinito
        trainlen = min(int(np.ceil(30*mpd)),maxtrainlen) if not trainlen else trainlen

        #Adiciona o trainlen_factor, que eh irrelevante para os parametros que estamos usando
        trainlen=int(trainlen*trainlen_factor)
        
        #define o raio inicial e final a serem usados no treinamento
        #nessa fase do algoritmo, esses valores sao grandes pois estamos focando em formar os 
        #grupos, sem necessariamente muita precisao. podemos buscar raios um pouco menores caso a
        #inicializacao tenha sido feita usando PCA
        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms/3.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/6.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms/8.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/4.) if not radiusfin else radiusfin

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def finetune_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Finetune training...")

        ms, mpd = self._calculate_ms_and_mpd()

        #mesma coisa que a fase de treinamento pesado, mas com um limite de vezes de execucao e
        #com um raio menor, para aperfeicoar ainda mais os grupos
        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms/12.)  if not radiusin else radiusin
            radiusfin = max(1, radiusin/25.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            trainlen = min(int(np.ceil(40*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, np.ceil(ms/8.)/4) if not radiusin else radiusin
            radiusfin = 1 if not radiusfin else radiusfin
        
        trainlen=int(trainlen_factor*trainlen)
            
        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    #metodo que faz os calculos dos blocos
    def _batchtrain(self, trainlen, radiusin, radiusfin, njob=1,
                    shared_memory=False):
        radius = np.linspace(radiusin, radiusfin, trainlen)

        #nao vamos usar memoria compartilhada
        if shared_memory:
            data = self._data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')

        else:
            data = self._data

        bmu = None

        #x2 é uma parte fixa da distancia euclidiana (x-y)^2 = x^2 +y^2 - 2xy 
        #que usamos na hora de calcular o erro de quantizacao
        #e para atualizar o bmu
        fixed_euclidean_x2 = np.einsum('ij,ij->i', data, data)

        logging.info(" radius_ini: %f , radius_final: %f, trainlen: %d\n" %
                     (radiusin, radiusfin, trainlen))

        #treina n vezes, onde n eh o tamanho do treinamento definido via parametro no inicio
        for i in range(trainlen):
            t1 = time()
            #calcula a vizinhanca passando os parametros de tamanho de vizinhanca definidos anteriormente
            neighborhood = self.neighborhood.calculate(
                self._distance_matrix, radius[i], self.codebook.nnodes)
            #encontra o bmu (best matching unit)
            bmu = self.find_bmu(data, njb=njob)
            #atualiza o codebook- que eh a matriz que guarda os valores de cada neuronio -
            #com a nova vizinhanca
            self.codebook.matrix = self.update_codebook_voronoi(data, bmu,
                                                                neighborhood)

            #calcula o erro de quantizacao e quanto tempo ele levou
            #o erro de quantizacao se dá pela media da soma do bmu com a distancia
            #euclidiana de cada dado, elevados ao quadrado
            qerror = (i + 1, round(time() - t1, 3),
                      np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)))

            logging.info(
                " epoch: %d ---> elapsed time:  %f, quantization error: %f\n" %
                qerror)
            if np.any(np.isnan(qerror)):
                logging.info("nan quantization error, exit train\n")
                            
        #atualiza o BMU somando seu valor antigo com as distancias euclidianas
        bmu[1] = np.sqrt(bmu[1] + fixed_euclidean_x2)
        self._bmu = bmu

    #funcao que encontra o BMU pra cada dado de forma paralela
    #recebe por parametro uma matriz em que as linhas sao os dados e as colunas dimensoes
    @timeit(logging.DEBUG)
    def find_bmu(self, input_matrix, njb=1, nth=1):
        #variavel que guarda a quantidade de dimensoes
        dlen = input_matrix.shape[0]
        #valcula as distancias euclidianas
        y2 = np.einsum('ij,ij->i', self.codebook.matrix, self.codebook.matrix)
        if njb == -1:
            njb = cpu_count()

        #toda essa estrutura foi criada para a paralizacao do processamento
        #a logica para de fato encontrar o BMU esta no metodo _chunk_based_bmu_find,
        #explicado mais abaixo
        pool = Pool(njb)
        chunk_bmu_finder = _chunk_based_bmu_find

        def row_chunk(part):
            return part * dlen // njb

        def col_chunk(part):
            return min((part+1)*dlen // njb, dlen)

        chunks = [input_matrix[row_chunk(i):col_chunk(i)] for i in range(njb)]
        b = pool.map(lambda chk: chunk_bmu_finder(chk, self.codebook.matrix, y2, nth=nth), chunks)
        pool.close()
        pool.join()
        bmu = np.asarray(list(itertools.chain(*b))).T
        del b
        return bmu

    """
    metodo para atualizar todos os neuronios que pertencem a vizinhanca do BMU.
    primeiro ele monta uma mini matriz com a vizinhanca de cada neuronio
    eh um metodo super eficiente, baseado na implementacao do algoritmo da SOM
    toolbox para matlab, feito pela Helsinky University
    parametros:
     - training_data: os dados no formato de sempre (m linhas sao dados, n colunas sao dimensoes)
     - bmu: o bmu pra cada dado. tem formato (2, m) e a primeira row guarda os indices dos BMU

    """
    @timeit(logging.DEBUG)
    def update_codebook_voronoi(self, training_data, bmu, neighborhood):
        #pega o indice do bmu
        row = bmu[0].astype(int)
        #cria uma coluna com valores de 0 ao numero de dados
        col = np.arange(self._dlen)
        #cria um array com um numero 1 para cada dado
        val = np.tile(1, self._dlen)
        #cria uma matriz esparsa com as dimensoes dos nos populada com os valores declarados acima 
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                       self._dlen))
        #multiplica a matriz esparsa pelos valores dos dados em treinamento
        S = P.dot(training_data)

        # a dimensao da vizinhanca eh de numero de nos x numero de nos 
        # a dimensao de S eh de numero de nos x dimensao dos dados
        # nominador tem dimensao numero de nos * dimensao
        nom = neighborhood.T.dot(S)
        #soma todos os valores da primeira dimensao da matriz esparsa e a transforma em uma coluna
        #com o mesmo tamanho do numero de nos
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)
        #multiplica entao esse valor com o valor da vizinhanca e volta o dado pra a estrutura de linha
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)
        #atualiza o codebook com os dados dividindo os dados calculados acima 
        new_codebook = np.divide(nom, denom)

        #e os retorna com limite de 6 casas decimais
        return np.around(new_codebook, decimals=6)

    def project_data(self, data):
        """
        Passa um dado novo numa SOM treinada e descobre seu lugar
        """
        #cria estrutura com os vizinhos, usando o metodo da classe neighbors
        #que pode ser visto na integra aqui:
        #https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/neighbors/classification.py#L23

        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        #cria linha com cada dimensao da estrutura de dados
        labels = np.arange(0, self.codebook.matrix.shape[0])
        #faz o fit usando a matriz de dados ja treinados como base para as labels
        clf.fit(self.codebook.matrix, labels)

        #normaliza os dados
        data = self._normalizer.normalize_by(self.data_raw, data)

        #descobre o lugar de cada dado
        return clf.predict(data)

    def predict_by(self, data, target, k=5, wt='distance'):
        # here it is assumed that target is the last column in the codebook
        # and data has dim-1 columns
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indX = ind[ind != target]
        x = self.codebook.matrix[:, indX]
        y = self.codebook.matrix[:, target]
        n_neighbors = k
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights=wt)
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indX]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indX], data)

        predicted_values = clf.predict(data)
        predicted_values = self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)
        return predicted_values

    def predict(self, x_test, k=5, wt='distance'):
        """
        Similar to SKlearn we assume that we have X_tr, Y_tr and X_test. Here
        it is assumed that target is the last column in the codebook and data
        has dim-1 columns

        :param x_test: input vector
        :param k: number of neighbors to use
        :param wt: method to use for the weights
            (more detail in KNeighborsRegressor docs)
        :returns: predicted values for the input data
        """
        target = self.data_raw.shape[1]-1
        x_train = self.codebook.matrix[:, :target]
        y_train = self.codebook.matrix[:, target]
        clf = neighbors.KNeighborsRegressor(k, weights=wt)
        clf.fit(x_train, y_train)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        x_test = self._normalizer.normalize_by(
            self.data_raw[:, :target], x_test)
        predicted_values = clf.predict(x_test)

        return self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)

    def find_k_nodes(self, data, k=5):
        from sklearn.neighbors import NearestNeighbors
        # we find the k most similar nodes to the input vector
        neighbor = NearestNeighbors(n_neighbors=k)
        neighbor.fit(self.codebook.matrix)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        return neighbor.kneighbors(
            self._normalizer.normalize_by(self.data_raw, data))

    def bmu_ind_to_xy(self, bmu_ind):
        """
        Translates a best matching unit index to the corresponding
        matrix x,y coordinates.

        :param bmu_ind: node index of the best matching unit
            (number of node from top left node)
        :returns: corresponding (x,y) coordinate
        """
        rows = self.codebook.mapsize[0]
        cols = self.codebook.mapsize[1]

        # bmu should be an integer between 0 to no_nodes
        out = np.zeros((bmu_ind.shape[0], 3))
        out[:, 2] = bmu_ind
        out[:, 0] = rows-1-bmu_ind / cols
        out[:, 0] = bmu_ind / cols
        out[:, 1] = bmu_ind % cols

        return out.astype(int)

    def cluster(self, n_clusters=8):
        import sklearn.cluster as clust
        cl_labels = clust.KMeans(n_clusters=n_clusters).fit_predict(
            self._normalizer.denormalize_by(self.data_raw,
                                            self.codebook.matrix))
        self.cluster_labels = cl_labels
        return cl_labels

    def predict_probability(self, data, target, k=5):
        """
        Predicts probability of the input data to be target

        :param data: data to predict, it is assumed that 'target' is the last
            column in the codebook, so data hould have dim-1 columns
        :param target: target to predict probability
        :param k: k parameter on KNeighborsRegressor
        :returns: probability of data been target
        """
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indx = ind[ind != target]
        x = self.codebook.matrix[:, indx]
        y = self.codebook.matrix[:, target]

        clf = neighbors.KNeighborsRegressor(k, weights='distance')
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indx]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indx], data)

        weights, ind = clf.kneighbors(data, n_neighbors=k,
                                      return_distance=True)
        weights = 1./weights
        sum_ = np.sum(weights, axis=1)
        weights = weights/sum_[:, np.newaxis]
        labels = np.sign(self.codebook.matrix[ind, target])
        labels[labels >= 0] = 1

        # for positives
        pos_prob = labels.copy()
        pos_prob[pos_prob < 0] = 0
        pos_prob *= weights
        pos_prob = np.sum(pos_prob, axis=1)[:, np.newaxis]

        # for negatives
        neg_prob = labels.copy()
        neg_prob[neg_prob > 0] = 0
        neg_prob = neg_prob * weights * -1
        neg_prob = np.sum(neg_prob, axis=1)[:, np.newaxis]

        return np.concatenate((pos_prob, neg_prob), axis=1)

    def node_activation(self, data, target=None, wt='distance'):
        weights, ind = None, None

        if not target:
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=self.codebook.nnodes)
            labels = np.arange(0, self.codebook.matrix.shape[0])
            clf.fit(self.codebook.matrix, labels)

            # The codebook values are all normalized
            # we can normalize the input data based on mean and std of
            # original data
            data = self._normalizer.normalize_by(self.data_raw, data)
            weights, ind = clf.kneighbors(data)

            # Softmax function
            weights = 1./weights

        return weights, ind

    def calculate_topographic_error(self):
        bmus1 = self.find_bmu(self.data_raw, njb=1, nth=1)
        bmus2 = self.find_bmu(self.data_raw, njb=1, nth=2)
        bmus_gap = np.abs((self.bmu_ind_to_xy(np.array(bmus1[0]))[:, 0:2] - self.bmu_ind_to_xy(np.array(bmus2[0]))[:, 0:2]).sum(axis=1))
        return np.mean(bmus_gap != 1)

    def calculate_map_size(self, lattice):
        """
        Calculates the optimal map size given a dataset using eigenvalues and eigenvectors. Matlab ported
        :lattice: 'rect' or 'hex'
        :return: map sizes
        """
        D = self.data_raw.copy()
        dlen = D.shape[0]
        dim = D.shape[1]
        munits = np.ceil(5 * (dlen ** 0.5))
        A = np.ndarray(shape=[dim, dim]) + np.Inf

        for i in range(dim):
            D[:, i] = D[:, i] - np.mean(D[np.isfinite(D[:, i]), i])

        for i in range(dim):
            for j in range(dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = sum(c) / len(c)
                A[j, i] = A[i, j]

        VS = np.linalg.eig(A)
        eigval = sorted(np.linalg.eig(A)[0])
        if eigval[-1] == 0 or eigval[-2] * munits < eigval[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigval[-1] / eigval[-2])

        if lattice == "rect":
            size1 = min(munits, round(np.sqrt(munits / ratio)))
        else:
            size1 = min(munits, round(np.sqrt(munits / ratio*np.sqrt(0.75))))

        size2 = round(munits / size1)

        return [int(size1), int(size2)]


# Since joblib.delayed uses Pickle, this method needs to be a top level
# method in order to be pickled
# Joblib is working on adding support for cloudpickle or dill which will allow
# class methods to be pickled
# when that that comes out we can move this to SOM class
def _chunk_based_bmu_find(input_matrix, codebook, y2, nth=1):
    """
    Finds the corresponding bmus to the input matrix.

    :param input_matrix: a matrix of input data, representing input vector as
                         rows, and vectors features/dimention as cols
                         when parallelizing the search, the input_matrix can be
                         a sub matrix from the bigger matrix
    :param codebook: matrix of weights to be used for the bmu search
    :param y2: <not sure>
    """
    dlen = input_matrix.shape[0]
    nnodes = codebook.shape[0]
    bmu = np.empty((dlen, 2))

    # It seems that small batches for large dlen is really faster:
    # that is because of ddata in loops and n_jobs. for large data it slows
    # down due to memory needs in parallel
    blen = min(50, dlen)
    i0 = 0

    while i0+1 <= dlen:
        low = i0
        high = min(dlen, i0+blen)
        i0 = i0+blen
        ddata = input_matrix[low:high+1]
        d = np.dot(codebook, ddata.T)
        d *= -2
        d += y2.reshape(nnodes, 1)
        bmu[low:high+1, 0] = np.argpartition(d, nth, axis=0)[nth-1]
        bmu[low:high+1, 1] = np.partition(d, nth, axis=0)[nth-1]
        del ddata

    return bmu



