import time
import torch
import torch.nn.functional as F
import dill as pickle
import pandas as pd
import torchtext
import os
import dill as pickle
import spacy
import re
from torch import nn
import math
import copy
import nltk
import numpy as np
import sys

device = 'cpu'
#%%
def remove_char(x):
    return re.sub(r'[^a-zA-Z\s]', '', x)
remove_char("I'm armin")
#%%
src_data = open('C:/arminpc/transform_data/english.txt').read().strip().split('\n')
trg_data = open('c:/arminpc/transform_data/french.txt', encoding="utf-8").read().strip().split('\n')
raw_data = {'src' : [line for line in src_data], 'trg': [line for line in trg_data]}
df = pd.DataFrame(raw_data, columns=["src", "trg"])
mask = (df['src'].str.count(' ') < 15) & (df['trg'].str.count(' ') < 15) # drop sentence if len is more than 80
df = df.loc[mask] # drop sentence if len is more than 80


df['src'] = df['src'].apply(lambda x: x.lower())
df['trg'] = df['trg'].apply(lambda x: x.lower())

df['src'] = df['src'].apply(remove_char)
df['trg'] = df['trg'].apply(remove_char)

df['src'] = df['src'].apply(nltk.word_tokenize)
df['trg'] = df['trg'].apply(nltk.word_tokenize)


def add_sos_eos(x):
    x.insert(0, 'sos')
    x.append('eos')
    return x

df['src'] = df['src'].apply(add_sos_eos)
df['trg'] = df['trg'].apply(add_sos_eos)

df = df[df['src'].map(len) < 17]
df = df[df['trg'].map(len) < 17]
#%%
df_copy = df.copy()
src_sentences = df_copy['src'].to_numpy() # Column to list
src_words = [j for i in src_sentences for j in i] # Split all words
src_words = list(set(src_words))
src_words = dict([(x, src_words.index(x)+1) for x in src_words])

trg_sentences = df_copy['trg'].to_numpy() # Column to list
trg_words = [j for i in trg_sentences for j in i] # Split all words
trg_words = list(set(trg_words))
trg_words = dict([(x, trg_words.index(x)+1) for x in trg_words])
#%%
def token_to_index_src(sentence):
    sentence_list = []
    for word in sentence:
        index = src_words[word]
        sentence_list.append(index)
    return sentence_list

def token_to_index_trg(sentence):
    sentence_list = []
    for word in sentence:
        index = trg_words[word]
        sentence_list.append(index)
    return sentence_list

df_copy['src_token'] = df_copy['src'].apply(token_to_index_src)
df_copy['trg_token'] = df_copy['trg'].apply(token_to_index_trg)

#%%
src_len = len(src_words)
trg_len = len(trg_words)
#%%
src_max_len = 0
trg_max_len = 0
    
for index, rows in df_copy.iterrows():
    src_token_index = rows['src_token']
    trg_token_index = rows['trg_token']
    
    if len(src_token_index) > src_max_len:
        src_max_len = len(src_token_index)
        
    if len(trg_token_index) > trg_max_len:
        trg_max_len = len(trg_token_index)
#%%
src_to_tensor = torchtext.transforms.ToTensor(dtype=torch.int64, padding_value=0)
trg_to_tensor = torchtext.transforms.ToTensor(dtype=torch.int64, padding_value=16)

class create_dataset(torch.utils.data.Dataset):
    def __init__(self, df, src_max_len, trg_max_len):
        self.df = df.copy()
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

    def __getitem__(self, idx):
        src = self.df['src_token'].iloc[idx]
        trg = self.df['trg_token'].iloc[idx]
        src = src + [0] * (src_max_len - len(src)) # https://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
        trg = trg + [0] * (trg_max_len - len(trg))
        src_tensor = src_to_tensor(src)
        trg_tensor = trg_to_tensor(trg)
        return [src_tensor, trg_tensor]
    
    def __len__(self):
        return len(self.df)
    
ds = create_dataset(df_copy, src_max_len, trg_max_len)
dl = torch.utils.data.DataLoader(ds, batch_size=16, drop_last=True, shuffle=False)

next(iter(dl))[0]

#%%
def Positional_Encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    return pe


class Encoder(nn.Module):
    def __init__ (self, N, H, src_vocab_size, d_model):
        super().__init__()
        self.pe = Positional_Encoding(16, d_model)
        self.d_model = d_model
        self.H = H
        self.N = N
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.LN = nn.LayerNorm(d_model)
        self.d_model_H = d_model/H
        self.q = nn.Linear(d_model, int(d_model/H))
        self.k = nn.Linear(d_model, int(d_model/H))
        self.v = nn.Linear(d_model, int(d_model/H))
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(d_model, d_model)
        self.FF1 = nn.Linear(d_model, 2048)
        self.FF2 = nn.Linear(2048, d_model)
        self.relu = nn.ReLU()
        
    def Multi_Head_Attn(self, qkv):
        q = self.q(qkv)
        k = self.k(qkv).transpose(1,2)
        v = self.v(qkv)
        q_k = torch.matmul(q, k) / math.sqrt(self.d_model/self.H)
        attn = self.softmax(q_k)
        attn = torch.matmul(attn, v)
        return attn
        
    def Encoder_Block(self, src):
        attn = self.Multi_Head_Attn(src)
        concated = attn
        for i in range(self.H-1):
            attn = self.Multi_Head_Attn(src)
            concated = torch.cat((concated, attn), 2)
        multi_attn = self.linear(concated)
        multi_attn = src + multi_attn
        multi_attn = self.LN(multi_attn)
        FF = self.FF1(multi_attn)
        FF = self.relu(FF)
        FF = self.FF2(FF)
        FF = FF + multi_attn
        FF = self.LN(FF)
        return FF
        
    def forward(self, src):
        src = self.encoder_embedding(src)
        src = src + self.pe
        for i in range(self.N):
            src = self.Encoder_Block(src)
        return src
        
        
class Decoder(nn.Module):
    def __init__ (self, N, H, trg_vocab_size, d_model):
        super().__init__()
        self.pe = Positional_Encoding(16, d_model)
        self.d_model = d_model
        self.H = H
        self.N = N
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.LN = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, int(d_model/H))
        self.k = nn.Linear(d_model, int(d_model/H))
        self.v = nn.Linear(d_model, int(d_model/H))
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(d_model, d_model)
        self.FF1 = nn.Linear(d_model, 2048)
        self.FF2 = nn.Linear(2048, d_model)
        self.relu = nn.ReLU()
        
    def Multi_Head_Attn_Mask(self, qkv, mask_index):
        q = self.q(qkv)
        k = self.k(qkv).transpose(1,2)
        v = self.v(qkv)
        q_k = torch.matmul(q, k) / math.sqrt(self.d_model/self.H)
        q_k[:,:,mask_index:] = -np.inf
        # print(q_k[0]); sys.exit()
        attn = self.softmax(q_k)
        
        attn = torch.matmul(attn, v)
        return attn
    
    def Multi_Head_Attn(self, q, output_encoder):
        q = self.q(q)
        k = self.k(output_encoder).transpose(1,2)
        v = self.v(output_encoder)
        q_k = torch.matmul(q, k) / math.sqrt(self.d_model/self.H)
        attn = self.softmax(q_k)
        attn = torch.matmul(attn, v)
        return attn
        
    def Decoder_Block(self, trg, output_encoder, mask_index):
        attn_mask = self.Multi_Head_Attn_Mask(trg, mask_index)
        concated = attn_mask
        for i in range(self.H-1):
            attn_mask = self.Multi_Head_Attn_Mask(trg, mask_index)
            concated = torch.cat((concated, attn_mask), 2)
        
        multi_attn_mask = self.linear(concated)
        multi_attn_mask = trg + multi_attn_mask
        multi_attn_mask = self.LN(multi_attn_mask)
        attn = self.Multi_Head_Attn(multi_attn_mask, output_encoder)
        concated = attn
        for i in range(self.H-1):
            attn = self.Multi_Head_Attn(multi_attn_mask, output_encoder)
            concated = torch.cat((concated, attn), 2)
        multi_attn = self.linear(concated)
        multi_attn = multi_attn + multi_attn_mask
        multi_attn = self.LN(multi_attn)
        FF = self.FF1(multi_attn)
        FF = self.relu(FF)
        FF = self.FF2(FF)
        FF = FF + multi_attn
        FF = self.LN(FF)
        return FF
        
    def forward(self, trg, output_encoder, mask_index):
        trg = self.decoder_embedding(trg)
        trg = trg + self.pe
        for i in range(self.N):
            trg = self.Decoder_Block(trg, output_encoder, mask_index)
            
        
        return trg
    

class Transformer(nn.Module):
    def __init__ (self, N, H, d_model, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.encoder = Encoder(N, H, src_vocab_size, d_model)
        self.decoder = Decoder(N, H, trg_vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, src, trg, mask_index, encoder_freez=True):
        output_encoder = self.encoder(src)
        output_decoder = self.decoder(trg, output_encoder, mask_index)
        
        output = self.linear(output_decoder)
        output = self.softmax(output)
        output = self.output(output)
        return output
        

model = Transformer(6, 8, 512, len(src_words), len(trg_words)).to(device).double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
loss_func = nn.CrossEntropyLoss()

for epoch in range(0,5):
    for i, batch in enumerate(dl):
        src = batch[0]
        trg = batch[1]
        optimizer.zero_grad()
        for j in range(1,len(trg)):
            with torch.set_grad_enabled(True):
                predicted = model(src, trg, mask_index=j, encoder_freez=False)
                predicted = torch.max(predicted, 0)[0]
                trg2 = trg[:,j]
                loss = loss_func(predicted, trg2)
                print(loss)
                if j == 10: sys.exit()
                loss.backward()
                optimizer.step()
                
        # print(predicted.size()); sys.exit()
        if i == 10: sys.exit()
        
#%%
z = torch.rand(3,3,3)
z[:,0,:] = np.inf
z
#%%
s = nn.Softmax(dim=0)
z = torch.rand(16,16,24)
zz = torch.rand(16,24,80)
z[:,0:2,:] = -np.inf
z = s(z)
# torch.matmul(z, zz).size()
z