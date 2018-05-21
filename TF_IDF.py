import numpy as np
import string

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

corpus = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

def limpeza():
    #Passa por todos os documentos e tenta remover as pontuações
    for i in range(0, len(corpus) - 1):
        #String.punctuation já possui um conjunto de pontuações bom
        #!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        corpus[i] = ''.join(char for char in corpus[i] if char not in string.punctuation)

def dicionario(corpus):
    #Poderia ser um dict também mas assim a iteração fica padronizada
    #E acessa-se com índices ao invés de termos
    dicionario = set()
    for doc in corpus:
        #Adiciona cada termo ao dicionario
        dicionario.update([termo for termo in doc.split()])
    return dicionario

def tf(termo, documento):
    #A função frequencia calcula o numero de vezes que o termo aparece no
    #documento
    aparicoes = frequencia(termo, documento)
    #conta a quantidade de palavras no documento
    palavras_documento = len(documento)
    #Calculo da frequencia do termo
    return (aparicoes/palavras_documento)

def frequencia(termo, documento):
    #Conta o numero de vezes que o termo aparece no documento
    return documento.split().count(termo)

def documentos_contendo(termo, corpus):
    #Conta o numero de documentos que possuem o termo
    cont = 0
    for doc in corpus:
        #itera sobre os documentos procurando o termo usando a função
        #frequencia
        if frequencia(termo, doc) > 0:
            cont += 1
    return cont

def idf(termo, corpus):
    #Conta o numero de documentos no corpus
    numero_documentos = len(corpus)
    #Conta o numero de documentos que possuem o termo
    doc_freq = documentos_contendo(termo, corpus)
    #Calcula o idf usando o log
    #baseado em https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    return np.log(numero_documentos/doc_freq)

def tf_idf(corpus, dicionario):
    vetor_tf = set()
    vetor_tfidf = set()
    #para cada termo do dicionario
    for termo in dicionario:
        #calcula o idf do termo em todo o corpus, é constante
        idf = idf(termo, corpus)
        for doc in corpus:
            #adiciona ao vetor_tf o tf para cada documento
            tf = tf(termo, doc)
            vetor_tfidf.update(tf*idf)

    return vetor_tfidf
#Provavelmente nem vamos usar esta função
def tf_idf_vetor(termo, corpus):
    vetor_tfidf = set()
    for doc in corpus:
        tf = tf(termo, doc)
        vetor_tfidf.update()
    return
#Usa a binaria do jeito forca bruta
def binaria(dicionario, corpus):
    for doc in corpus:
        bin_vetor_aux = [frequencia(termo, doc) for termo in dicionario]
    bin_vetor = set()
    for term in bin_vetor_aux:
        if(bin_vetor_aux[term] > 0):
            bin_vetor.update(1)
        else:
            bin_vetor.update(0)
    return bin_vetor
#cria o vetor binario usando uma funcao lambda que encapsula tudo que a
#funcao acima faz, mas em uma linha (not sure if works)
def binaria2(dicionario, corpus):
    bin_vector = [(lambda termo: 1 if frequencia(termo,doc) > 0 else 0 for termo in dicionario)]
    return bin_vector
#outra funcao para a binaria para a construcao da matriz
def binaria3(termo, doc):
    bin_vector = [(lambda termo: 1 if frequencia(termo,doc) > 0 else 0 for termo in dicionario)]
    return bin_vector
#Executa a limpeza das pontuacoes
limpeza()
#cria o dicionario limpo
dicionario = dicionario(corpus)
#cria o vetor tfidf
vetor_tfidf = tf_idf(corpus, dicionario)
#cria a matriz de tf
doc_tf_matriz = []
for doc in corpus:
    tf_vetor = [tf(termo, doc) for termo in dicionario]
    doc_tf_matriz.append(tf_vetor)

#Multiplica a matriz tf pelo vetor tfidf
#Funciona pois o numero de colunas da matriz é igual ao numero de linhas
matriz_tfidf = doc_tf_matriz.dot(vetor_tfidf)
#Podemos remover o print da matriz depois
print(matriz_tfidf)

bin_matriz = []
for doc in corpus:
    bin_vet = [binaria3(termo, doc) for termo in dicionario]
    bin_matriz.append(bin_vet)
#podemos remover esse print também
print(bin_matriz)