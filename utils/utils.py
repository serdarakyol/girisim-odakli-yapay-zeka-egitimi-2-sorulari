import re
import time
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

def load_model(path):
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model


def download_stopwords(url = 'https://raw.githubusercontent.com/sgsinclair/trombone/master/src/main/resources/org/voyanttools/trombone/keywords/stop.tr.turkish-lucene.txt'):
    data = pd.read_csv(url)
    stpwrds = set(data[2:].iloc[:, 0].values)
    return stpwrds


# Cümle vektorizasyonu fonksiyonu
def vectorize(cumle, stpwrds, word_vectors):
    # İlk önce cümledeki noktalama işaretlerini kaldırınız. 
    cumle = re.sub(r'[^\w\s]', ' ', cumle)
    # clean digits for model vectorize correct
    cumle = re.sub(r'[0-9]+', ' ', cumle)

    # Cümleyi kelimeler haline getiriniz.
    words = cumle.split(" ")

    # Stopwords listesinde yer almayan her kelimenin vektörünü bulup ortalamasını alın.
    clean_words = [word for word in words if word not in stpwrds]

    counter = 0
    init_vectors = np.zeros((400,))
    for word in clean_words:
        word = word.lower().strip()
        if (len(word) > 1):
            w_vector = word_vectors.get_vector(word)
            init_vectors += w_vector
            counter += 1

    return init_vectors / counter
