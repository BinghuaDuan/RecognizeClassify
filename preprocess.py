import random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

tag2label_classify = {"POS": 0, "NEG": 1, "CET": 2}

def read_corpus(corpus_path,model_type):
    """
    读取语料，返回样本列表。
    :param corpus_path:语料路径
    :param model_type:模型类型
    :return: data_recognize:功能识别模型的语料 type:[([],[]),...,([],[])]
    data_classify:情感分类模型 type:[([],str),...,([],str)]
    """
    # data = []
    # with open(corpus_path, encoding='utf-8') as fr:
    #     lines = fr.readlines()
    # sent_, tag_ = [], []
    # for (i, line) in enumerate(lines):
        # if line != '\n':
        #     if line[0] != '。':
        #         [char, label] = line.strip().split()
        #
        #         sent_.append(char)
        #         tag_.append(label)
        #     else:
        #         data.append((sent_, tag_))
        #         sent_, tag_ = [], []
    data_recognize = []
    data_classify = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for (i, line) in enumerate(lines):
        line = line.strip('\n')
        sentiment = ['POS', 'NEG', 'CET']
        if line not in sentiment:
            [char, label] = line.split(' ')
            sent_.append(char)
            tag_.append(label)
        else:
            data_recognize.append((sent_, tag_))
            data_classify.append((sent_, line))
            sent_, tag_ = [], []


    if model_type == 'recognize':
        return data_recognize
    else:
        return data_classify


def sentence2id(sent, word2id):
    """
    将样本中的每个词转换成词典中的id.
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def random_embedding(word2id, embedding_dim):
    """
    将词转换成嵌入学习的随机向量。
    :param word2id:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat




#-------------------------------------------------------------------------------------------
def pad_sequences(sequences, pad_mark=0):
    """
    将读取的句子填充成大小一样的序列
    :param sequences:
    :param pad_mark:
    :return:seq_list:按最长样本的长度填充样本，seq_len_list:每条样本的真实长度
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, word2id, tag2label, shuffle=False):
    """
    每次读取一个batch_size的样本进行训练或测试。
    :param data: 如果是识别模型，则typeof(data)=[([],[])],如果是分类模型，则typeof(data)=[([],str)]
    :param batch_size:
    :param word2id:
    :param tag2label:
    :param shuffle:
    :return: seqs:batch_size个样本，type：[[],[]]
             labels:batch_size个样本对应的标签，model_type=”recognize",type:[[],[]]
             model_type="classify",type:[]
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, word2id)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

