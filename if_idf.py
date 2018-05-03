import pandas as pd

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

def remove_pont():
    for i in range(0, len(all_documents)-1):
        for j in range(0, len(all_documents[i])-1):
            if (all_documents[i][j] == ","):
                all_documents[i].replace(all_documents[i][j], "")
            if (all_documents[i][j] == "."):
                all_documents[i].replace(all_documents[i][j], "")
            if (all_documents[i][j] == "?"):
                all_documents[i].replace(all_documents[i][j], "")
            if (all_documents[i][j] == ":"):
                all_documents[i].replace(all_documents[i][j], "")
            if (all_documents[i][j] == "'"):
                all_documents[i].replace(all_documents[i][j], "")
        f.write(all_documents[i][j])

def remove():
    for i in range(0, len(all_documents) - 1):
        print(''.join( c for c in all_documents[i] if  c not in '?:,.' ))
        f.write(all_documents[i])

f = open('documentos.txt','w')
remove()
#all_documents[0].replace("China", "Dougras")
#print(all_documents[0])
f.close()