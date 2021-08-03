#%%
import random
import mxnet as mx
from mxnet import gluon, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()

#%%
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5,d=16,
                d_prime=4, drop_ratio=0.05,**kwargs):
        super(Caser, self).__init__(**kwargs)
    


    def forward(self, user_id, seq, item_id):
#%%
import torch 
# %%
