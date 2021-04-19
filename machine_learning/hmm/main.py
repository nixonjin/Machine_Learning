#%%
import pickle
from os.path import join
import os,sys
base_dir = os.path.dirname(__file__)

sys.path.append(base_dir)
from .hmm import HMM
from .metric import Metrics
# down load data from https://github.com/luopeixiang/named_entity_recognition

def build_corpus(split, make_vocab=True, data_dir="./data"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name,"wb") as f:
        pickle.dump(model,f)

def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_O=False):
    """训练并评估hmm模型"""
    # 训练hmm模型
    train_word_lists,train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)
    save_model(hmm_model,"data/hmm.pkl")

    # 评估hmm模型
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)
    
    metrics = Metrics(test_tag_lists,pred_tag_lists,remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

#%%
"""训练模型，评估结果"""

# 读取数据
print("读取数据...")
train_word_lists, train_tag_lists, word2id, tag2id = \
    build_corpus("train")
dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

# 训练评估ｈｍｍ模型
print("正在训练评估HMM模型...")
hmm_pred = hmm_train_eval(
    (train_word_lists, train_tag_lists),
    (test_word_lists, test_tag_lists),
    word2id,
    tag2id
)
# %%
