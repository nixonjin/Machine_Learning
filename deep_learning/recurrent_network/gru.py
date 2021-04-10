#%%
from d2l import torch as d2l
import torch
import math
from torch import nn
from torch.nn import functional as F
#%%
gru_layer = nn.GRU(num_inputs, num_hiddens)
#%%
#@save
class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

#%%
batch_size, num_steps = 32, 35
# train_iter: [(X,Y),...,]   X:(batch_size,num_steps), Y:(batch_size, num_steps)
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

#%%
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device))
    
    W_xz, W_hz, b_z = three() # Update gate parameters
    W_xr, W_hr, b_r = three() # Reset gate parameters
    W_xh, W_hh, b_h = three() # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens),device=device),)


def predict(prefix, num_preds, model, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(torch.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    # H : (batch_size, num_hiddens)
    H, = state
    outputs = []
    # inputs: (num_steps, batch_size, vocab_size)
    # X : (batch_size, vocab_size)
    for X in inputs:
        # Z : (batch_size, num_hiddens)  Z is the weight that decide whether H to be updated 
        # comes from H or comes from H_tilda
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)

        # R : (batch_size, num_hiddens)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)

        # H_tilda : (batch_size, num_hiddens)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh)+b_h)

        # H : (batch_size, num_hiddens)
        H = Z * H + (1 - Z) * H_tilda

        # Y : (batch_size, num_outputs) = (batch_size,vocab_size)
        Y= H @ W_hq + b_q
        outputs.append(Y)
    
    # outputs: (num_steps, batch_size, num_outputs) -> (num_steps * batch_size, num_outputs)
    return torch.cat(outputs, dim=0), (H,)

def grad_clipping(params, theta):
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#%%
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
#%%
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
loss = torch.nn.CrossEntropyLoss()
params = get_params(vocab_size,num_hiddens,device)
optimizer = torch.optim.SGD(params,lr)
animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
for epoch in range(num_epochs):
    timer = d2l.Timer()
    metric = [0.0,0.0]
    # X, Y : (batch_size, num_steps)
    for X, Y in train_iter:
        state = init_gru_state(batch_size,num_hiddens,device)
        # Y: (batch_size, num_steps) - > (num_steps * batch_size)
        y = Y.T.reshape(-1)
        # X: (batch_size, num_steps) - > (num_steps, batch_size, vocab_size)
        X = F.one_hot(X.T, vocab_size).type(torch.float32)
        X, y = X.to(device), y.to(device)
        y_hat, state = gru(X, state, params)

        # The `input` is expected to contain raw, unnormalized scores for each class.
        # Examples::
        # |  
        # |      >>> loss = nn.CrossEntropyLoss()
        # |      >>> input = torch.randn(3, 5, requires_grad=True)
        # |      >>> target = torch.empty(3, dtype=torch.long).random_(5)
        # |      >>> output = loss(input, target)
        # |      >>> output.backward()
        l = loss(y_hat, y.long()).mean()
        optimizer.zero_grad()
        l.backward()
        grad_clipping(params,1)
        optimizer.step()
        metric[0] += l*len(y)
        metric[1] += len(y)

    ppl, speed =  math.exp(metric[0]/metric[1]), metric[1]/timer.stop()
    if (epoch + 1) % 10 == 0:
        animator.add(epoch + 1, [ppl])

print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')

# %%
