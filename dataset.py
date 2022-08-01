# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:38:57 2022

@author: 이현호
"""

import torch 
import numpy as np
import pandas as pd  
import random
import os 

from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer,AddedToken

def seed_everything(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    


class CustomDataset(Dataset):
    def __init__(self, dataset, option):
        self.dataset = dataset 
        self.option = option
        self.pretrain = "klue/bert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain)
        self.tokenizer.add_special_tokens({'bos_token':AddedToken("[BOS]", lstrip=True)})
    
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:3].values
        encoder_text = row[0]
        decoder_text = row[1]
    
        encoder_inputs = self.tokenizer(
            encoder_text,
            return_tensors='pt',
            max_length=512,
            pad_to_max_length=True,
            add_special_tokens=False,
            truncation = True)
        
        decoder_inputs = self.tokenizer(
            decoder_text,
            return_tensors='pt',
            max_length=512,
            pad_to_max_length=True,
            add_special_tokens=False,
            truncation = True)
        
        encoder_input_ids = encoder_inputs['input_ids'][0]
        #encoder_attention_mask = encoder_inputs['attention_mask'][0]
        decoder_input_ids = decoder_inputs['input_ids'][0]
        label = row[2]
       
        
        if self.option =='train':
            label = row[2]
            return encoder_input_ids,decoder_input_ids, label
        
        return encoder_input_ids, decoder_input_ids,label
    


def load_data(train='ratings_train.txt', test='ratings_test.txt'):
    train.to_csv('./textdata/train_data.csv', index=False)
    test.to_csv('./textdata/test_data.csv', index=False)
    train = train[['document','decode_document' ,'label']]
    test = test[['document','decode_document' ,'label']]


    return train, test


if __name__=="__main__":
    torch.manual_seed(777)
    random.seed(777)
    np.random.seed(777)

    train = pd.read_csv('./textdata/ratings_train.txt',sep='\t').dropna()
    test = pd.read_csv('./textdata/''ratings_test.txt',sep='\t').dropna()
    train['decode_document'] = "[BOS]" + train['document']
    test['decode_document'] = "[BOS]" + test['document']
    train = train[['document','decode_document' ,'label']]
    test = test[['document','decode_document' ,'label']]
    train.to_csv('./textdata/train_data.csv', index=False)
    test.to_csv('./textdata/test_data.csv', index=False)
    train = train[['document','decode_document' ,'label']]
    test = test[['document','decode_document' ,'label']]

    train_dataset = CustomDataset(train, 'train')
    test_dataset = CustomDataset(test, 'test')



    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


    