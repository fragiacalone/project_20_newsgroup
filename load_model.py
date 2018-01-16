from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, Conv1D
from keras.models import load_model
import pickle

def load_cnn_news():
    model = load_model('weights/model_news.h5')
    print("Definizione del modello terminata")
    model.load_weights('weights/w_news.h5')
    print("Caricamento dei pesi terminato")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    print("Compile del modello terminato")
    return model

def load_cnn_news_labels():
    return {'0': 'alt.atheism:               ',
         '1': 'comp.graphics:             ',
         '2': 'comp.os.ms-windows.misc:   ',
         '3': 'comp.sys.ibm.pc.hardware:  ',
         '4': 'comp.sys.mac.hardware:     ',
         '5': 'comp.windows.x:            ',
         '6': 'misc.forsale:              ',
         '7': 'rec.autos:                 ',
         '8': 'rec.motorcycles:           ',
         '9': 'rec.sport.baseball:        ',
         '10': 'rec.sport.hockey:          ',
         '11': 'sci.crypt:                 ',
         '12': 'sci.electronics:           ',
         '13': 'sci.med:                   ',
         '14': 'sci.space:                 ',
         '15': 'soc.religion.christian:    ',
         '16': 'talk.politics.misc:        ',
         '17': 'talk.politics.guns:        ',
         '18': 'talk.politics.mideast:     ',
         '19': 'talk.religion.misc:        '}

def load_cnn_news_word_index():
    with open('./word_index.pkl', 'rb') as f:
        return pickle.load(f)    
    
def load_keywords():
    with open('./keywords_tfidf.pkl', 'rb') as f:
        return pickle.load(f)    
    