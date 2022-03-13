#%%
from d2l import torch as d2l
import torch
import torch.nn as nn

#%%
T =1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time,[x],'time','x',xlim=[1,1000],figsize=(6,3))
# %%
