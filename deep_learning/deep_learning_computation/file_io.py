#%%
import torch
from torch import nn
from torch.nn import functional as F 

#%%
x = torch.arange(4)
torch.save(x,'x-file')

# %%
x2 = torch.load("x-file")
x2
# %%
