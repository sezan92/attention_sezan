
# coding: utf-8

# In[1]:


# In[importing libraries]
import keras
from keras.layers import Input,Dense,LSTM,Embedding,Bidirectional,multiply,RepeatVector,Flatten
from keras.models import Model
from keras.utils import plot_model
import text_preprocess_utils as tpu
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
# In[2]:


import pickle


# Japanese

# In[3]:


source = pickle.load(open('/data/ben_eng_source_vocab_fh.pkl','rb'))
target = pickle.load(open('/data/ben_eng_target_vocab_fh.pkl','rb'))
df= pd.read_csv('/data/ben.txtTest.csv')

# In[4]:


source_vocab = source[0] #first one is vocabulary list
source_i2w = source[1] # second one is vocabulary word to index
source_w2i = source[2] # third one is vocabulary index to word
target_vocab = target[0] #first one is vocabulary list
target_i2w = target[1] # second one is vocabulary word to index
target_w2i = target[2] # third one is vocabulary index to word
source_vocab_size = len(source_vocab)+2
target_vocab_size = len(target_vocab)+2
n_a = 32
n_s =64

# In[5]:


model = model_from_json(open('/data/ben_model_12_words.json').read())
model.load_weights('/data/attention_ben_12_words_eng.final.hdf5')


# In[6]:


model.summary()


# In[8]:


tp = tpu.text_prep('/data/ben.txt',limit=20000)
tp.show()


# In[9]:


source_input,input_starter,output_encoded = tp.preprocess(12)


# In[133]:


a1 =source_input[-10]

c = output_encoded[-10]



# In[150]:

init_state = np.zeros((source_input.shape[0],n_s))
init_hidden = np.zeros((source_input.shape[0],n_s))
a2= init_state[0].reshape((-1,n_s))
a3 = init_hidden[0].reshape((-1,n_s))

# In[151]:




# In[152]:


a1 = a1.reshape((-1,12))


# In[153]:


pred = model.predict(x=[a1,a2,a3])


# In[172]:



# In[190]:


def decode(source,pred):
    pred_words=[]
    for prediction in pred:
        indice = np.argmax(prediction)
        #print('indice '+str(indice))
        if indice==0:
            continue
        elif target_i2w[indice] =='<e>':
            break
        else:
            pred_words.append(target_i2w[indice])
    source_words = [source_i2w[word] for word in source]
    source_sen = ' '.join(source_words)
    pred_tran = ' '.join(pred_words)
    print("English: "+source_sen)
    print("Translated "+pred_tran)
    return pred_tran


# In[]
output_list =[]
# In[191]:
for index,row in df.iterrows():
    sentence = row['Source']
    words = sentence.lower().split()
    encoded_sentence = [source_w2i[word] for word in words]
    encoded_sentence_pad = pad_sequences([encoded_sentence],maxlen=12,padding='post',truncating='post')
    pred =model.predict(x=[encoded_sentence_pad,a2,a3])
    pred_sentence=decode(encoded_sentence,pred)
    output_list.append(pred_sentence)
df['Prediction']=output_list
df.to_csv('/output/Prediction.csv')

