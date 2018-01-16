from matplotlib import pyplot as plt
from keras.preprocessing import text
from keras import backend as K
import numpy as np#
import tensorflow as tf
import re, csv, operator

K._LEARNING_PHASE = tf.constant(0)
MAX_NB_WORDS = 30000
MAX_SEQUENCE_LENGTH = 600

def get_prediction (text, model, word_index):
    x = []
    text = re.sub('([.,!?()\"\'])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    for word in text.split(' '):
        word = word.lower()
        try:
            index = word_index[word]
            if index >= MAX_NB_WORDS:
                x.append(0)
            else:
                x.append(index)
        except:
            x.append(0)
    while len(x) < MAX_SEQUENCE_LENGTH:
        x.insert(0, 0)
    x = np.asmatrix(x)
    prediction = model.predict(x) 
    return prediction

def get_predictions(original_text, perturbated, filename, m):
    
    model = m[0]
    labels = m[1]
    word_index = m[2]
    
    out_file = open("results/predictions_"+filename+".txt", "w") 
    
    
    original_prediction = get_prediction(original_text, model, word_index)
    out_file.write('Original text: ' + original_text + ' \n\n')
    for i in range(0, len(labels)):
        percentage = float(int(original_prediction[0][i]*10000))/100
        #if percentage > 0.9:
        out_file.write(labels[str(i)]  + str(percentage) + ' %\n')
    
    labels_features = {}
    for i in range(0, len(labels)):
        labels_features[labels[str(i)]] = {}
    
    
    for p in perturbated:
        written = False
        prediction = get_prediction(perturbated[p], model, word_index)
        delta= original_prediction - prediction
        ratio = original_prediction/prediction
        for i in range(0, len(labels)):
            or_percentage = float(int(original_prediction[0][i]*10000))/100
            percentage = float(int(prediction[0][i]*10000))/100
            if prediction[0][i] != 0:
                r_value = float(int(ratio[0][i]*100))/100
            else:
                r_value = 99999

            if (ratio[0][i] > 1.10 or ratio[0][i] < 0.90) and (delta[0][i] > 0.03 or delta[0][i] < -0.03):
                if written == False:
                    #out_file.write('\n\n - - - - - - - - - - - - \n\n')
                    #out_file.write('Feature \''+p+'\':\n\n')
                    written = True
                
                #out_file.write(labels[str(i)] + str(percentage) + ' -> '+ str(or_percentage) + ' %   -   Ratio: '+ str(r_value) + ' %  -  ')       
                labels_features[labels[str(i)]][(p+':').ljust(20, ' ') + str(percentage) + ' -> '+ str(or_percentage) + ' %'] = r_value
                #if ratio[0][i] < 0.9: 
                    #out_file.write('Negative impact\n')
                #else:    
                    #out_file.write('Positive impact\n')
    
    out_file.write("\n\n\n")
    for k in labels_features:
        if len(labels_features[k]) != 0:
            out_file.write(k + "\n")
            fdict = labels_features[k]
            while len(fdict) != 0:
                km = max(fdict, key=fdict.get)

                vm = fdict[km]

                fdict.pop(km, None)
                out_file.write("\t" + str(vm) + ' - '+ km + '\n')
            out_file.write("\n\n")
        
    out_file.close()  
    print('Predizioni terminate')