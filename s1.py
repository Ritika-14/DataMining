import collections
import itertools
import scipy
import sklearn
import math
from math import sqrt
from sklearn import metrics, neighbors
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import numpy as np
import pandas as pd
import re
import nltk
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
	
fname = r"C:\Users\Ritika\Downloads\1612289430_9552789_train.dat"
'''  The train data file is read and the reviews are stored in train, while the classifications are stored in y'''
stopword_set = set(stopwords.words("english"))
train = []
y = []
with open(fname, 'r', encoding="utf8") as infile:
  for line in infile.readlines():
    train.append(line[3:])
    y.append(line[0:2])
''' Data preprocessing is done by removing  HTML tags, #EOF characters, removing speacial charcters and punctuation marks,
    and each word is stemmed'''
train = [i.replace('<br>', '').replace('-',' ').replace('<br />', '').replace('#EOF','') for i in train]
text_result = []

for ele in train[0:18750]:
  s =''
  ele = " ".join([i for i in re.sub(r'[^a-zA-Z0-9\s]', "", ele).split() if i not in stopword_set])
  #print(ele)
  tokens = word_tokenize(ele)
  snowball_stemmer = SnowballStemmer('english')
  #print(tokens)
  for t in tokens:
      s += " "+ snowball_stemmer.stem(t)
      
  text_result.append(s)


fname1 = r"C:\Users\Ritika\Downloads\1612289431_1872573_test.dat"
test = []
with open(fname1, 'r', encoding="utf8") as infile:
  for line in infile.readlines():
    test.append(line[0:])
#print(len(test))
test = [i.replace('<br>', '').replace('-',' ').replace('<br />', '').replace('#EOF','') for i in test]
text_result1 = []
for ele in train[18750:]:
  s =''
  ele = " ".join([i for i in re.sub(r'[^a-zA-Z0-9\s]', "", ele).split() if i not in stopword_set])
  #print(ele)
  tokens = word_tokenize(ele)
  snowball_stemmer = SnowballStemmer('english')
  #print(tokens)
  for t in tokens:
      s += " "+ snowball_stemmer.stem(t) 
  text_result1.append(s)
'''
CountVectorizer is used to store the counts of each word in the document in the form of a 2-D matrix,
dimensionality reduction is performed to keep 65000 features, and ngram = (1,2) is used for best accuracy results'''
vectorizer = CountVectorizer(max_features = 65000, binary = True, analyzer='word', encoding='utf8', input='content',stop_words='english', lowercase = True, ngram_range=(1,2))
#text_result = ["the cat sat on my face", "the dog sat on my bed"]
#text_result1 = ["the cat is very funny"]
S1 = (vectorizer.fit_transform(text_result + text_result1)).toarray()
print("done 1..")
'''
len1 = len(text_result)
idf = []
IDF values are computed by finding the log(total number of documents/number of documents a word appers in) by calculating the non-zero elements
    of each column
counts = np.count_nonzero(S1[0:25000][0:], axis=0)
for c in counts:
  if c!=0:
    idf.append(math.log10(len1/c))
  else:
    idf.append(0.0)
print("done 1..")
#print(idf)


#print(terms)

vocab = set()
for document in text_result:
  word = get_ngrams(document,2)
  vocab |= set(word)
#print("vocab")
#print(vocab)
print("done 2 ..")
wordsets = [ frozenset(get_ngrams(document,2)+get_ngrams(document,1)) for document in text_result ]
print(wordsets)
results = {}
for word in terms:
    count = sum( 1 for s in wordsets if word in s )
    results[word ]=count
print(results)
print("done .. 3")
idf = []
for term in terms:
  if term not in vocab:
    idf.append(0.0)
  else:
    idf.append(math.log10(len1/results[term]))

print("done 4..")

'''
''' The TFIDF matrix is computed by first normlizing the TF matrix for train and test and  multiplying it with the IDF of train set'''
'''sum_of_rows = S1[0:25000][0:].sum(axis=1)
normalized_array =  S1[0:25000][0:] / sum_of_rows[:, np.newaxis]
tfidf_train = (normalized_array*idf)
print(tfidf_train )
#np.savetxt('tfidf_train.txt',tfidf_train)
print("done 5..")

sum_of_rows = S1[25000:][0:].sum(axis=1)
normalized_array =  S1[25000:][0:] / sum_of_rows[:, np.newaxis]
tfidf_test = (normalized_array*idf)
print(tfidf_test )
#np.savetxt('tfidf_test.txt', tfidf_test)
print("done 6..")

#print(tfidf)

n_components = 20000
pca = TruncatedSVD(n_components, n_iter=7, random_state=42)
S1 = pca.fit_transform(S1)

print(S.shape)

#bag_of_words_test = vectorizer.transform(text_result1).toarray()
S1 = csr_matrix(vectorizer.transform(text_result1)).toarray()
#pca1 = TruncatedSVD(n_components)
#S1 = pca.fit_transform(S1)
print(S1)
#a = (scipy.spatial.distance.cdist(tfidf_test[0:1000][0:], tfidf_train, 'euclidean'))
print(a)
print("done 7..")

#tfidf_train =  tfidf_train.astype(np.float32)
#tfidf_test = tfidf_test.astype(np.float32)


 The KNN Algorithm is run by finding the cosine similarity and locating the k nearest neighbors of each test review with the train matrix
'''
#a = sklearn.metrics.pairwise.cosine_similarity(S1[18750:][0:], S1[0:18750][0:], dense_output=False)
a = sklearn.metrics.pairwise_distances(S1[18750:][0:], S1[0:18750][0:], metric='euclidean')
#a = sklearn.metrics.pairwise.cosine_similarity(tfidf_test[0:5000][0:], tfidf_train, dense_output=False)
#np.savetxt('a.txt',a)
print("done 7..")
knn_indices= []
for ls in a: #dists is the 2D vector of distances
    sorted_distances_indices = np.argsort(ls) #Getting a sorted list of indices of all distances in ls with the smallest distance's index at 0th position
    #knn_indices = []
    indexes = (list(itertools.islice(sorted_distances_indices,199)))
    knn_l = []
    for i in indexes:
        l = y[i]
        knn_l.append(l)
    knn_indices.append(knn_l)
print("done 8..")
p = []
with open('predictions_final.txt', 'w', encoding="utf8") as infile:
  for k in knn_indices:
    prediction = max(set(k), key=k.count)
    p.append(prediction)
    #infile.write(prediction + "\n")
print(sklearn.metrics.accuracy_score(y[18750:], p))
	





