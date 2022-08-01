# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:51:11 2022

@author: 이현호
"""


import json 
import torch 


class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
        
        

config = Config({
    "n_enc_vocab": 32001,
    "n_dec_vocab": 32001,
    "n_enc_seq": 512,
    "n_dec_seq": 512,
    "n_layer": 6,
    "d_hidn": 256,
    "i_pad": 0,
    "d_ff": 1024,
    "n_head": 4,
    "d_head": 64,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-12,
    "config.n_output":2,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
})
print(config)