import matplotlib.pyplot as plt
from matplotlib import style
style.use('presentation')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [1,3],
              [8,9],
              [0,3],
              [5,4],
              [6,4],])

colors = 10*["g","r","c","b","k"]

k=2
tol=0.001
max_iter=300

def fit(data):
    #Inicializa os centroides
    centroids = {}
    #Posiciona os centroides em cima dos dados (Primeiro tipo de inicialização)
    for i in range(k):
        centroids[i] = data[i]
    #Aqui começam as iterações
    for i in range(max_iter):
        #Em cada rodada os classificadores começam vazios
        classifications = {}

        for i in range(k):
            #"Limpa" os classificadores que possam existir
            classifications[i] = []
        #Roda pelo dados conhecidos e atribui ao centroide mais proximo
        for featureset in data:
            #Compara as distâncias entre os centróides
            distances = [np.linalg.norm(featureset-centroids[centroid]) for centroid in centroids]
            #Aloca ao classificador o que possuir menor distância
            classification = distances.index(min(distances))
            #Atualiza a posição do classificador
            classifications[classification].append(featureset)
        #Cria um dict dos centroides para uso posterior
        prev_centroids = dict(centroids)
        #Atualiza os centroides usando o valor médio com uso de pesos no eixo 0, por isso usa o average e não o mean
        for classification in classifications:
            #Na primeira rodada não faz isso
            centroids[classification] = np.average(classifications[classification],axis=0)
        #Se convergiu já deixa marcado para testar condição de saída
        optimized = True
        #Teste da movimentação dos centroides
        for c in centroids:
            #Pega o centroide original
            original_centroid = prev_centroids[c]
            #Pega o centroide atual
            current_centroid = centroids[c]
            #Subtrai o atual do original e checa se bate com a tolerância
            if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > tol:
                #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                optimized = False
        #Se a movimentação está dentro da tolerância, para
        if optimized:
            break
#Pega os dados dos centroides já treinados, usa a mesma a logica inicial
def predict(data):
    #Compara a distância entre cada centróide
    distances = [np.linalg.norm(data-centroids[centroid]) for centroid in centroids]
    #Pega o classificador com a menor distância
    classification = distances.index(min(distances))
    return classification
#Chama o classificador
clf.fit(X)
#busca entre os centroides para imprimir (Apenas para visualização)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)
#Percorre entre os classificadores para pegar a cor de cada centroide
for classification in clf.classifications:
    #Define a cor de cada cluster baseado no classificador e no vetor de cores
    color = colors[classification]
    #Imprime os dados baseado no seu cluster/centroide e cor
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()