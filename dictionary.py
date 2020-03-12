##描述：该模块主要是和字典相关的操作，包含两个函数：
#build_dictionary():根据语料创建字典。
#read_dictionary():读取已经创建好的字典。
import pickle, os
from preprocess import read_corpus
def build_dictionary(dict_path, corpus_path, min_count=1):
    """
    功能描述：根据语料创建字典
    :param dict_path: 创建后字典的存储位置
    :param corpus_path:语料的位置
    :param min_count:字的最小频次
    :return:
    """
    data = read_corpus(corpus_path,model_type='classify')
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print('len(word2id)={}'.format(len(word2id)))
    with open(dict_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def read_dictionary(dict_path):
    """
    读取字典。
    :param dict_path:
    :return:word2id type:dict
    """
    dict_path = os.path.join(dict_path)
    with open(dict_path, 'rb') as fr:
        word2id = pickle.load(fr)
    # print('vocab_size:', len(word2id))
    return word2id

if __name__ == '__main__':

    dict_path = os.path.join(os.getcwd(), 'data_path', 'word2id.pkl')
    corpus_path = os.path.join(os.getcwd(), 'data_path', 'train_data')

    build_dictionary(dict_path=dict_path, corpus_path = corpus_path, min_count=0)
   #  word2id = read_dictionary(dict_path)
   #  train_data = read_corpus(corpus_path=corpus_path,model_type='classify')
   #  transform(train_data, word2id)

    # train_data_path = os.path.join(os.getcwd(), 'data_path', 'train_data')
    # train_data = read_corpus(corpus_path=train_data_path)
    # print('len(train_data)={}'.format(len(train_data)))
    # s_len = [len(s[0]) for s in train_data]
    # print(max(s_len))

