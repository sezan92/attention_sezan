#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:17:36 2018

@author: sezan92
"""
# In[1] '''Importing packages'''
# python 3.6.3
import pandas as pd
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm as tq

# In[2] '''reading file'''
class text_prep():
    def __init__(self,filename,limit=10000,head=False,full=False):
        self.filename =filename
        print('Reading file.......')
        self.df_actual = pd.read_csv(self.filename,delimiter='\t',header=None,names=['Source','Target'])
        self.df_test =self.df_actual.sample(500)
        self.df_actual.drop(self.df_test.index)
        self.df_test.to_csv('/output/%sTest.csv'%(filename[8:]))
        self.limit=limit
        if not full:
            print("Taking %d data"%(self.limit))
            self.df = self.df_actual.head(self.limit) if head else self.df_actual.tail(self.limit)
# In[3] '''showing first five rows'''
    def show(self,num=5):
        '''Shows a number of rows ,by default 5'''
        print('first %d rows...'%num)
        print(self.df.head(num))
    def encode_text(self,ls,w2i,max_length):
        """ function for encoding texts
        ls = text list
        w2i = dictionary for word to index
        max_length = maximum possible length
        """
        encoded = [np.array([w2i[word] for word in sentence.lower().split()]) for sentence in ls]
        return encoded
    def maxlen(self,source_list,default=0):
        """method for setting maximum length.
        it will give default as maximum length if it is given"""
        if default==0:
            return int(np.max([len(source_sen.split()) for source_sen in source_list]))
        else:
            return default
    def preprocess(self,max_length=0):
        """Preprocess data
        output: 
        source_input: encoded source input
        target_input: encoded target input
        output: output words encoded"""
        print("Adding start and end tag....")
        for index,row in self.df.iterrows():
            row['Target']='<s> '+row['Target']+' <e>'
            
        '''Reading source and target Text'''
        print('Reading source and target Text....')
        self.source = self.df['Source'] # Reading source texts
        self.source_list = self.source.values.tolist() #Converting datatargetme into list
        self.source_str = ' '.join(self.source_list) #Converting list to string
        self.source_str = self.source_str.lower()
        self.target = self.df['Target'] # Reading Target texts
        self.target_list = self.target.values.tolist() #Converting datatargetme into list
        self.target_str = ' '.join(self.target_list) #Converting list to string
        self.target_str = self.target_str.lower()

        '''Tokenizing words'''
        print('Tokenizing words....')
        self.source_vocab = sorted(set(self.source_str.split()))
        self.source_vocab_size=len(self.source_vocab)+1
        print('Vocabulary Size %d'%self.source_vocab_size)

        self.target_vocab = sorted(set(self.target_str.split()))
        self.target_vocab_size=len(self.target_vocab)+1
        print('Vocabulary Size %d...'%self.target_vocab_size)


        self.source_i2w = dict((i+1,c)for i,c in enumerate(self.source_vocab))
        self.source_w2i = dict((c,i+1)for i,c in enumerate(self.source_vocab))

        self.target_i2w = dict((i+1,c)for i,c in enumerate(self.target_vocab))
        self.target_w2i = dict((c,i+1)for i,c in enumerate(self.target_vocab))
        '''Saving vocabulary'''
        '''Pickling vocabulary....'''
        #print('Pickling vocabulary....')
        #pickle.dump(file=open('fra-eng/target_vocab.pkl','wb'),obj=(self.target_vocab,self.target_i2w,self.target_w2i))
        #pickle.dump(file=open('fra-eng/source_vocab.pkl','wb'),obj=(self.source_vocab,self.source_i2w,self.source_w2i))

        ''' Maximum length for both'''
        print(' Maximum length for both')
        self.source_max_length = self.maxlen(self.source_list,max_length)
        self.target_max_length = self.maxlen(self.target_list,max_length)
        print("Maximum length for Source %d"%self.source_max_length)
        print("Maximum length for Target language %d"%self.target_max_length)
        ''' Average length for both'''
        print(' Average length for both')
        self.source_ave_length = int(np.mean([len(source_sen.split()) for source_sen in self.source_list]))
        self.target_ave_length = int(np.mean([len(target_sen.split()) for target_sen in self.target_list]))
        print("Average length for Source %d"%self.source_ave_length)
        print("Average length for Target language %d"%self.target_ave_length)
        ''' function Encode Texts'''
        """Encoding"""
        print("Encoding.....")
        self.source_encoded = self.encode_text(self.source_list,self.source_w2i,self.source_max_length)
        self.target_encoded = self.encode_text(self.target_list,self.target_w2i,self.target_max_length)
        "Preprocessing Data"
        print("Preprocessing data....")
        """source_encoded_extended=[]
        target_encoded_extended=[]
        output_encoded =[]
        for ls in tq(self.target_encoded):
            for word in ls:
                source_encoded_extended.append(self.source_encoded[self.target_encoded.index(ls)])
                target_encoded_extended.append(ls[:ls.index(word)+1])
                output_encoded.append(ls[ls.index(word)])
        source_input= pad_sequences(source_encoded_extended,maxlen=self.source_max_length)
        target_input = pad_sequences(target_encoded_extended,maxlen=self.target_max_length)
        return source_input,target_input,output_encoded"""
        input_starter=[]
        output_encoded=[]        
        for sentence in tq(self.target_encoded):
            input_starter.append([sentence[0]])
            output_encoded.append(sentence[1:])
        source_input= pad_sequences(self.source_encoded,maxlen=self.source_max_length,truncating='post',padding='post')
        input_starter = np.array(input_starter)
        output_encoded = pad_sequences(output_encoded,maxlen=self.target_max_length,truncating='post',padding='post') 
        #output_encoded = [pad_sequences([output_en],maxlen=self.target_max_length,truncating='post',padding='post') for output_en in output_encoded]
        return source_input,input_starter,output_encoded
    def get_df(self):
        """Getting the dataframe of translation"""
        return self.df
    def text2csv(self,name='Target'):
        """saving the text dataframe into csv file"""
        self.df_actual.to_csv('/output/%s Text With Translation.csv'%name)
    def get_length(self):
        """ Returns maximum lengths of the languages"""
        return self.source_max_length,self.target_max_length
    def get_vocab(self):
        """Get vocabulary
        target: Target vocabulary, target vocabulary dict for index to word and word to index
        source: Source Vocabulary , source vocabulary dict for index to word and word to index"""
        
        target = (self.target_vocab,self.target_i2w,self.target_w2i)
        source =(self.source_vocab,self.source_i2w,self.source_w2i)
        return source,target
