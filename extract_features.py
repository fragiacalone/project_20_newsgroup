import nltk.data, string
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def sentence_tokenizer(text):
    #nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    features = tokenizer.tokenize(text)
    return features

def textiling_tokenizer(text):
    nltk.download('stopwords')
    tokenizer = nltk.tokenize.TextTilingTokenizer()
    features = tokenizer.tokenize(text)
    return features

def get_perturbated_input(features):
    perturbated = []
    for i in range(-1, len(features)):
        string = ''
        for j in range(0, len(features)):
            if j != i or i == -1:
                string = string + features[j] + ' '
        perturbated.append(str(i)+': '+string)
    return perturbated

punctuation_map = dict((ord(char), None) for char in string.punctuation)
stemmer = nltk.stem.snowball.EnglishStemmer()

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(punctuation_map)))


def get_clusters(sentences, filename):
    vectorizer = TfidfVectorizer(tokenizer=normalize)
    tf_idf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = (tf_idf_matrix * tf_idf_matrix.T).A
    affinity_propagation = AffinityPropagation(affinity="precomputed", damping=0.5)
    affinity_propagation.fit(similarity_matrix)

    labels = affinity_propagation.labels_
    cluster_centers = affinity_propagation.cluster_centers_indices_

    tagged_sentences = zip(sentences, labels)
    clusters = {}
    out_file = open("results/clusters_"+filename+".txt", "w") 
    for sentence, cluster_id in tagged_sentences:
        #clusters.setdefault(cluster_id, []).append(sentence)
        clusters[sentence] = cluster_id
        out_file.write(str(clusters[sentence]) + ": " + str(sentence) + "\n\n")
        
    return clusters

def get_perturbated(sentences, labelled):
    perturbated = {}
    features = {}
    lablist = list(set(labelled.values()))
    for label in lablist:
        ptext = ''
        ftext = ''
        for sentence in sentences:
            if labelled[sentence] != label:
                ptext = ptext + ' ' + sentence
            else: ftext = ftext + ' ' + sentence
        perturbated[label] = ptext
        features[label] = ftext
    return [perturbated, features]
    