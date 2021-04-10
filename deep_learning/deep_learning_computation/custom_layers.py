#%%
import torch
from torch import nn
import torch.nn.functional as F

# Layers without Parameters
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X-X.mean()

#%%
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

#%%  
net = MyLinear(2,20)
X = torch.randn(size=(2,))
Y = net(X)

#%%
torch.save(net.state_dict(),'mlp.params')

# %%
clone = MyLinear(2,20)
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
# %%
