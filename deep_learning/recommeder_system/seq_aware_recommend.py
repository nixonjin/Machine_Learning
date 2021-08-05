#%%
import random
import torch
from torch import nn 


#%%
class Caser(nn.Module):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4):
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.


    def forward(self,user_id,item_id,seq):
        pass

#%%
num_users=10
num_items=10
P=nn.Embedding(num_users,32)
Q=nn.Embedding(num_items,32)
#%%
seq = torch.as_tensor([1,2,3,4,5,6])
temp = Q(seq)
# %%
temp = temp.unsqueeze(1)
temp.shape
# %%
