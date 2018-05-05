#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:31:21 2018

@author: sezan92
"""
# In[importing libraries]
import keras
from keras.layers import Input,Permute,Dense,LSTM,Embedding,Bidirectional,multiply,RepeatVector,Flatten,Activation
from keras.models import Model
from keras.activations import softmax
from keras.utils import plot_model
import text_preprocess_utils_fh as tpu
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import pickle
import os
# In[Getting Text encoded]
tp = tpu.text_prep('/data/ben.txt',limit=20000)
tp.show()
tp.text2csv("Bengali")
# In[preprocess]
max_words = 12
source_input,input_starter,output_encoded = tp.preprocess(max_words)
# In[get vocabulary]
source_words,target= tp.get_vocab()
source_vocab = source_words[0] #first one is vocabulary list
source_i2w = source_words[1] # second one is vocabulary word to index
source_w2i = source_words[2] # third one is vocabulary index to word
target_vocab = target[0] #first one is vocabulary list
target_i2w = target[1] # second one is vocabulary word to index
target_w2i = target[2] # third one is vocabulary index to word
source_vocab_size = len(source_vocab)+2
target_vocab_size = len(target_vocab)+2
# In[timesteps i.e. words per sentance]
source_timesteps = source_input.shape[1]
target_timesteps = output_encoded.shape[1]
n_a= 32
n_s =64
# In[parameters and Hyperparameters]
em_shape=100
batch_size=64
epochs = 100
# In[saving vocabulary]
pickle.dump(file=open('/output/ben_eng_source_vocab_fh.pkl','wb'),obj=source_words)
pickle.dump(file=open('/output/ben_eng_target_vocab_fh.pkl','wb'),obj=target)

# In[model]
source = Input(shape=(source_timesteps,),name='source') #Source sequence
print(source.shape)
source_emb = Embedding(input_dim=source_vocab_size ,output_dim=100,name='source_embedding',mask_zero=True)(source) #Embedding for source sequence
print(source_emb.shape)
h_source = Bidirectional(LSTM(32,return_sequences=True,name='h_s'))(source_emb) #Hidden state of source sequence
print(h_source.shape)
# In[]
initial_hidden = Input(shape=(n_s,),name='hidden_target') #Initial hidden state of target , we will give input <s> as starting of the sequence
init_state_att=initial_hidden
print(init_state_att.shape)
init_hid = Input(shape=(n_s,),name='cell_target')
print(init_hid.shape)
init_hid_att=init_hid
init_state_att_repeat = RepeatVector(source_timesteps)(init_state_att)
print(init_state_att_repeat.shape)
output=[] #Output empty list

# In[test]
for _ in range(target_timesteps): # For loop for manually looping through sequences
    merged = multiply([init_state_att_repeat,h_source]) #Dot product as of h_t and h_s
    score = Dense(1,activation='tanh')(merged) # tanh(h_txh_s)
    attention_prob = Dense(1,activation='softmax')(score) #prob = softmax(tanh(h_txh_s))
    context = multiply([h_source,attention_prob]) #context = probxh_source
    init_state_att,_,init_hid_att = LSTM(64,return_state=True)(context,initial_state=[init_state_att,init_hid_att]) #hidden state of next word of target
    init_state_att_repeat = RepeatVector(source_timesteps)(init_state_att) #making it 3D by repeat vector
    #context = merge([attention_prob,h_source],mode='mul',name='context_vector')
    prediction = Dense(target_vocab_size,activation='softmax',kernel_regularizer=l2())(init_state_att) #predicting next word
    output.append(prediction) #appending to output list
# In[]
model =Model(inputs=[source,initial_hidden,init_hid],outputs=output)
# In[]
model.compile(optimizer=Adam(0.008),loss='sparse_categorical_crossentropy')
model.summary()

if os.path.exists('/data/attention_ben_%d_words_eng.best.hdf5'%max_words):
    model.load_weights('/data/attention_ben_%d_words_eng.best.hdf5'%max_words)

filepath="/output/attention_ben_%d_words_eng.best.hdf5"%max_words
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
# In[plot model]
#plot_model(model,to_file="output/initial.png",show_shapes=True,show_layer_names=True)
# In[save model]
json_file = model.to_json()
with open('/output/ben_model_%d_words.json'%max_words,'w') as file:
    file.write(json_file)
# In[split model]
#model.fit()
x_train1,x_test1,y_train,y_test= train_test_split(
        source_input,
        output_encoded)
x_train2 = np.zeros((x_train1.shape[0],n_s))
x_train3 = np.zeros((x_train1.shape[0],n_s))

x_test2 = np.zeros((x_test1.shape[0],n_s))
x_test3 = np.zeros((x_test1.shape[0],n_s))

# In[fit]
model.fit(x=[np.array(x_train1),
             x_train2,x_train3],
    y=list(y_train.swapaxes(0,1)),
    validation_data=([np.array(x_test1),
             x_test2,x_test3],
    list(y_test.swapaxes(0,1))),
    batch_size=batch_size,epochs=epochs,
    callbacks=callbacks_list)
# In[save]
model.save_weights("/output/attention_ben_%d_words_eng.final.hdf5"%max_words)