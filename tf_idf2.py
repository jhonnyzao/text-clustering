import numpy as np
import math

#Baseado em https://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/tfidf.html

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

def remove():
    for i in range(0, len(all_documents) - 1):
        all_documents[i] = ''.join( c for c in all_documents[i] if  c not in '?:,.')

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon


def tf(term, document):
    return freq(term, document)


def freq(term, document):
    return document.split().count(term)

def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)


remove()
vocabulary = build_lexicon(all_documents)

doc_term_matrix = []
print('Vetor de vocabulario [' + ', '.join(list(vocabulary)) + ']')
for doc in all_documents:
    print('O documento é "' + doc + '"')
    tf_vector = [tf(word, doc) for word in vocabulary]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    print('O vetor de termFreq para o documento %d é [%s]' % ((all_documents.index(doc) + 1), tf_vector_string))
    doc_term_matrix.append(tf_vector)

    # here's a test: why did I wrap mydoclist.index(doc)+1 in parens?  it returns an int...
    # try it!  type(mydoclist.index(doc) + 1)

print('Matriz de termos do documento: ')
print(doc_term_matrix)

doc_term_matrix_l2 = []
for vec in doc_term_matrix:
    doc_term_matrix_l2.append(l2_normalizer(vec))

print ('matriz de termos antigas: ')
print (np.matrix(doc_term_matrix))
print ('matriz de termos normalizada:')
print (np.matrix(doc_term_matrix_l2))

my_idf_vector = [idf(word, all_documents) for word in vocabulary]

print ('O vetor de vocabulário é [' + ', '.join(list(vocabulary)) + ']')
print ('O vetor de tf-idf é [' + ', '.join(format(freq, 'f') for freq in my_idf_vector) + ']')