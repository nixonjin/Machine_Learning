#%%
import random
import torch
import pandas as pd
import os
from torch import nn
from d2l import mxnet as d2l
from torch._C import dtype
from torch.utils.data import Dataset
#%%
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
#%%
def split_data_ml100k(data, num_users, num_items, test_ratio=0.1):
    """spilt the dataset into train set and test set in seq aware mode
    parameters:
        data: a dataframe whose columns are ['user_id', 'item_id', 'rating', 'timestamp']
        ...
    """
    train_items, test_items, train_list = {}, {}, []
    for line in data.itertuples():
        u, i, rating, time = line[1], line[2], line[3], line[4]
        train_items.setdefault(u,[]).append((u,i,rating,time))
        if u not in test_items or test_items[u][-1] < time:
            # test items are all the last item that a user bought, so-called seq-aware model
            test_items[u] = (i, rating, time)
    # user id is start from 1
    for u in range(1, num_users + 1):
        train_list.extend(sorted(train_items[u], key=lambda k:k[3]))
    test_data = [(key, *value) for key, value in test_items.items()]
    # exclude the items that are exited in test_data 
    train_data = [item for item in train_list if item not in test_items]
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    # train data, test data: u, i, rating, time

def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    """
        transform the data into another form, that is, from a u-i-rating-time record into 
        seperate arrays, users, items, scores, interaction matix
    """
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        # user id, item_id are counted from 1
        user_index, item_index = int(line[0]-1), int(line[1]-1)
        users.append(user_index)
        items.append(item_index)
        score = int(line[3]) if feedback=='explicit' else 1
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter

class SeqDataset(Dataset):
    def __init__(self, user_ids, item_ids, seq_len, num_users, num_items,
                candidates):
        """
        paramenters:
            user_ids: all user ids
            item_ids: all item ids
            seq_len: the length of sequence in seq-aware mode
            ...
            candidates: a Dict whose key is user_id and value is a list containing 
            item ids that are bought by the user.
        return: Null
        """
        user_ids, item_ids = torch.as_tensor(user_ids), torch.as_tensor(item_ids)
        sort_idx = sorted(range(len(user_ids)), key=lambda k: user_ids[k])
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        # temp is used to record how many items that a user owns
        temp = {}
        for i, u_id in enumerate(u_ids):

    
    def __len__(self):
        pass

    def __getitem__(self, index):
        """
        parameter:
            index
        return:
            user_id, item sequence, target item, negative item
            
            训练时分别以(user_id, items sequence, target item)和
            (user_id, item sequence, negative item)作为输入, 模型会输出他们的相应概率值，最后
            使用BPR Loss（Bayesian Personalized Ranking Loss）来更新模型参数
        """

        return super().__getitem__(index)
        
#%%
class Caser(nn.Module):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4):
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)


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
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = torch.randn(20, 16, 50, 100)
#%%
output = m(input)
# %%
d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices,
                  d2l.evaluate_ranking, candidates, eval_step=1)
#%%
import random
import mxnet as mx
from mxnet import gluon, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
#%%
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully-connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(
            sorted(range(len(user_ids)), key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(
            sum([
                c - L if c >= L + 1 else 1
                for c in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, -step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
#%%
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users, num_items,
                            candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
#%%
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer, {
    "learning_rate": lr,
    'wd': wd})

d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices,
                  d2l.evaluate_ranking, candidates, eval_step=1)
# %%
