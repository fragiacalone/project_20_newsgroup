import os, sys, pickle, re, numpy, csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
import nltk
import pickle
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

MAX_NB_WORDS = 4000
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')


if not os.path.exists('corpus.pkl'):    #il corpus è stato già processato e salvato
    with open('corpus.pkl', 'rb') as f:
        texts = pickle.load(f) 
else:                                   # il corpus non è stato ancora processato
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    
    print('Processing text dataset')
    
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    t = t.replace('\n', '')
                    t = re.sub(r'[^\w\s]','',t)
                    tokens = word_tokenize(t)
                    filtered_tokens = [word.lower() for word in tokens if word not in stop_words]
                    lmtzr = WordNetLemmatizer()
                    lems = [lmtzr.lemmatize(t) for t in filtered_tokens]
                    texts.append(" ".join(lems))
                    f.close()
                    labels.append(label_id)
    
    print('Found %s texts.' % len(texts))
    with open("corpus.pkl", "wb") as corpus_file:
        pickle.dump(texts, corpus_file)
   

count_vect = CountVectorizer(stop_words= "english")
X_train_counts = count_vect.fit_transform(texts)

vocabulary = count_vect.vocabulary_

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

svd = TruncatedSVD(n_components = 10)
svd_matrix_terms = svd.fit_transform(X_train_tf.T)
#svd_matrix_docs = svd.fit_transform(X_train_tf)

csvReport = open('svd.csv', 'w', newline='')  
for v in vocabulary:
    if(max(svd_matrix_terms[vocabulary[v]]) > 0.5):  
        csvReport.write(v + ";")  
        for sv in svd_matrix_terms[vocabulary[v]]:       
            csvReport.write(str(sv) + ";")
        csvReport.write("\n")
csvReport.close()

#----------------------------------------------------------------------------------------------------

vcos = cosine_similarity( svd_matrix_terms[vocabulary["satellite"]] ,svd_matrix_terms)
vlist = []
for v in vocabulary.keys():
    k = vocabulary[v]
    vlist.append((vcos[0][k], v))
    
csvReport = open('word.csv', 'w', newline='')
for v in vlist:
    csvReport.write(str(v[0]) + ";" + str(v[1]) + "\n") 
csvReport.close() 

#---------------------------------------------------------------------------------------------------
