mydoclist = ['Julie loves me more than Linda loves me',
'Jane likes me more than Julie loves me',
'He likes basketball more than baseball']

#Baseado em http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/
#mydoclist = ['sun sky bright', 'sun sun bright']

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

from collections import Counter

remove()
for doc in all_documents:
    tf = Counter()
    for word in doc.split():
        tf[word] +=1
    print (tf.items())