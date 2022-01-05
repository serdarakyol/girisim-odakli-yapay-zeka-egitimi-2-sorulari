import streamlit as st
from utils.utils import *


sentences = ["Fenerbahçe-Çaykur Rizespor maçına Atilla Szalai damgası! Vitor Pereira'nın o kararı",
             "Fenerbahçe, Süper Lig'in 15. haftasında Çaykur Rizespor'u 4-0 mağlup ederek maç fazlasıyla ikinci sıraya yükseldi",
             'Galatasaray Lazio hazırlıklarına başladı',
             'Son dakika haberi: 5 Aralık corona virüsü tablosu ve vaka sayısı Sağlık Bakanlığı tarafından açıklandı!',
             'Bakan Akar, Katar Savunma Bakanı Atiyye ile görüştü']


input_text = st.text_area("Liste halinde verilerinizi giriniz lütfen", sentences)

# load model and stopwords
model = load_model("data/trmodel")
stopwords = download_stopwords()

# Vectorize fonksiyonu kullanarak elimizdeki cümlelerin vektörlerini elde ediniz. 
all_vectors = []
for sent in sentences:
    all_vectors.append(vectorize(cumle=sent, stpwrds=stopwords, word_vectors=model))

# Kosinüs benzerliği ile bu 5 cümle için 5x5'lik bir benzerlik matrisi oluşturunuz.
similarities = []
for vec in all_vectors:
    temp = []
    for vec2 in all_vectors:
        temp.append(cosine_similarity(vec.reshape(1, -1), vec2.reshape(1, -1)))
    similarities.append(temp)

del all_vectors

# Matris satır toplamını alınız, skor 1x5 vektörü oluşturunuz.
final_similarities = []
for similarity in similarities:
    sum = 0
    for sim in similarity:
        sum += sim
    final_similarities.append(sum)

max_score_indexes = []
for i in range(len(final_similarities)):
    max_item = max(final_similarities)
    index = final_similarities.index(max_item)
    max_score_indexes.append(index + i)
    final_similarities.remove(max_item)
    if len(max_score_indexes) == 2:
        break

del final_similarities, similarities

final_list = []
for id in max_score_indexes:
    final_list.append(sentences[id])

st.text_area("Sonuç", final_list, key="predicted_list")
