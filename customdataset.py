# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 12:05:59 2022

@author: 이현호
"""

import pandas as pd 

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer,AddedToken



def load_data(train='ratings_train.txt', test='ratings_test.txt'):
    train = pd.read_csv('./textdata/'+ train,sep='\t').dropna()
    test = pd.read_csv('./textdata/'+ test,sep='\t').dropna()
    train['decode_document'] = "[BOS]" + train['document']
    test['decode_document'] = "[BOS]" + test['document']
    train = train[['document','decode_document' ,'label']]
    test = test[['document','decode_document' ,'label']]
    train.to_csv('./textdata/train_data.csv', index=False)
    test.to_csv('./textdata/test_data.csv', index=False)
    train = train[['document','decode_document' ,'label']]
    test = test[['document','decode_document' ,'label']]


    return train, test


train,test=load_data()

#%%

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
    
    

def get_loader(batch_size, model_name,num_workers):
  
    
    train_loader = DataLoader(dataset=CustomDataset(train, 'train'),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    test_loader = DataLoader(dataset=CustomDataset(test, 'test'),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, test_loader

def get_dist_loader(batch_size,model_name, num_workers):

    
    train_dataset = CustomDataset(train, 'train')
    test_dataset = CustomDataset(test, 'train')
    
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler,
                              pin_memory=True,
                              batch_size=batch_size,
                              shuffle=None,
                              num_workers=num_workers)
    
    test_loader = DataLoader(dataset=test_dataset,
                            sampler=test_sampler,
                            pin_memory=True,
                            batch_size=batch_size,
                            shuffle=None,
                            num_workers=num_workers)
    
    return train_loader, test_loader, train_sampler, test_sampler




