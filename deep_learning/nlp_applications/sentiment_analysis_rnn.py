#%%
from d2l import torch as d2l
import torch
from torch import nn
import os 

#%%@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')
#%%
data_dir = d2l.download_extract('aclImdb', 'aclImdb')

#%%
def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos','neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                    label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n','')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels
#%%
train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:',y,'review',x[0:60])

# %%
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

d2l.set_figsize()
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0,1000,50))

#%%
num_steps = 500
train_features = torch.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)

#%%
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)
for X, y in train_iter:
    print('X:', X.shape, ',y:', y.shape)
    break
print('#batches:', len(train_iter))

#%%
def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos','neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                    label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n','')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels
def load_data_imdb(batch_size, num_steps=500):
    data_dir = d2l.download_extract('aclImdb','aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

#%%
from d2l import torch as d2l 
import torch 
from torch import nn 
#%%
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')
batch_size = 64
train_iter, test_iter, vocab = load_data_imdb(batch_size)

#%%
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(num_hiddens*4, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0],outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
#%%
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

#%%
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights)

#%%
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
#%%
embeds.shape
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
#%%
from sklearn.metrics import accuracy_score
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
net.train()
ls = []
accuracy = []
#%%
for epoch in range(num_epochs):
    print("executing epoch %s"%epoch)
    i = 0
    for X, y in train_iter:
        trainer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        trainer.step()
        ls.append(l)
        accuracy.append(accuracy_score(y, y_hat.argmax(1)))
        i += 1
        if i % 10 == 0:
            print("loss: %s, accuracy %s"%(l,accuracy[-1]))
#%%
import matplotlib.pyplot  as plt 
plt.figure()
plt.plot(list(range(len(ls))), ls)
plt.plot(list(range(len(accuracy))), accuracy)
# %%
