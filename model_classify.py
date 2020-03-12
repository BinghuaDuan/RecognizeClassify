from keras import layers, models
import numpy as np
from preprocess import sentence2id, tag2label_classify,read_corpus
from dictionary import read_dictionary
from keras.preprocessing.sequence import pad_sequences
class TwoLayerLSTM:
    def __init__(self,batch_size, epoch_nums, dict_length, embedding_dim, save_path,time_step):
        self.batch_size = batch_size
        self.epoch_nums = epoch_nums
        self.dict_length = dict_length
        self.embedding_dim = embedding_dim
        self.hidden_nums = 128
        self.class_nums = 3
        self.model = None
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'sgd'
        self.metrics = ['accuracy']
        self.save_path = save_path
        self.time_step = time_step


    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Embedding(input_dim=self.dict_length,
                                        output_dim=self.embedding_dim,
                                        input_length=self.time_step))
        self.model.add(layers.LSTM(units=self.hidden_nums, dropout=0.1, return_sequences=True))
        self.model.add(layers.LSTM(units=self.hidden_nums, return_sequences=True))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=self.class_nums, activation='softmax'))



    def train(self,train_data,word2id):
        print("train:==================================")
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.summary()
        (X_train, Y_train) = self.transform(train_data, word2id)
        self.model.fit(X_train, Y_train, epochs=self.epoch_nums, batch_size=self.batch_size)
        self.model.save(self.save_path)

    def test(self, test_data, wod2id):
        print("test:===================================")
        (X_test, Y_test) = self.transform(test_data,wod2id)
        self.model.evaluate(X_test, Y_test, self.batch_size)

    def transform(self, train_or_test_data, word2id):
        """
        将数据转换成X_data,Y_data的形式
        :param train_or_test_data: 训练集或测试集 type[([],str)]
        :return: X_data,Y_data type:np.array(())
        """
        # ids_seq = []
        X_data = []
        Y_data = []
        for(seq, tag) in train_or_test_data:
            seq_id = sentence2id(seq, word2id)
            label_id = tag2label_classify[tag]
            one_hot = list([0]*self.class_nums)
            one_hot[label_id] = 1
            # ids_seq.append(seq_id)
            X_data.append(seq_id)
            Y_data.append(one_hot)
        #按time_step的长度来填充样本
        # max_len = self.time_step
        # pad_mark = 0
        # for ids in ids_seq:
        #     ids_ = ids[:max_len] + [pad_mark] * max(max_len - len(ids), 0)
        #     X_data.append(ids_)

        X_data = pad_sequences(X_data, maxlen=self.time_step)
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)
        return (X_data,Y_data)


def predict(mode_path, sentence, word2id, time_step):
    """
    :param mode_path: 训练好的模型的存储位置
    :param sentence: 需要预测的句子
    :param word2id: 字典
    :return:
    """
    seqs = list(sentence)
    seqsid = sentence2id(seqs, word2id) #需要填充至time_step
    # time_step = 45
    # pad_mark = 0
    # seqs_id = seqsid[:time_step] + [pad_mark] * max(time_step - len(seqsid), 0)
    seqs_id = np.array([seqsid])
    seqs_id = pad_sequences(seqs_id, maxlen=time_step)
    model = models.load_model(mode_path)
    result = model.predict(seqs_id)
    result = result.tolist()[0]
    label = result.index(max(result))
    label2tag = {0: "新增", 1: "减少", 2: "不变"}

    return label2tag[label]


import os
if __name__ == '__main__':
    dictionary_path = os.path.join(os.getcwd(), "data_path", "word2id.pkl")
    word2id = read_dictionary(dictionary_path)
    corpus_path = os.path.join(os.getcwd(), "data_path", "train_data")
    train_data = read_corpus(corpus_path, model_type='classify')
    save_path = os.path.join(os.getcwd(),"data_path_save", "TwoLayerLstm.model")
    time_step = 60
    batch_size = 8
    epoch_nums = 40
    embedding_dim = 300
    # Model = TwoLayerLSTM(batch_size,epoch_nums,len(word2id), embedding_dim, save_path,time_step)
    # Model.build_model()
    # Model.train(train_data, word2id)
    sentence1 = '为了使老人能够更好地享受服务，针对每个老人的不同需求，恭和养老院最新推出了个性化特殊饮食。'
    sentence2 = '福提园引入北京师范大学心理专业资源，开展形式各异的心理辅导、认知训练、延缓认知衰退等服务。'
    label = predict(save_path, sentence2, word2id, time_step)
    print(label)




# dict_length = 1000
# args_embedding_dim = 300
# batch_size = 3
# time_step = 20
# X_train = np.random.randint(0,1000,size=(batch_size * 5, time_step))
# Y_train = np.random.randint(0,2,size=(batch_size * 5, 2))
# X_test = np.random.randint(0,1000,size=(batch_size * 2, time_step))
# Y_test = np.random.randint(0,2,size=(batch_size * 2, 2))
# model = models.Sequential()
# model.add(layers.Embedding(input_dim=dict_length, output_dim=args_embedding_dim))
# model.add(layers.LSTM(units=64, dropout=0.1, return_sequences=True))
# model.add(layers.LSTM(units=64, return_sequences=True))
# model.add(layers.Flatten())
# model.add(layers.Dense(units=2, activation='softmax'))
# model.add(layers.Activation('softmax'))categorical_crossentropy
# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
# model.summary()
# model.fit(X_train, Y_train,epochs=2,batch_size=batch_size)
# loss_and_metric = model.evaluate(X_test, Y_test, batch_size)
# demo = np.random.randint(0,1000,size=(2,time_step))
# result = model.predict(demo)
# model.save('TwoLayerLstm.model')
# print("#demo")
# print(demo)
# print("#result")
# print(result)



