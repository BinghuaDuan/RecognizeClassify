from model_classify import predict
from dictionary import read_dictionary
import os
#全局设置
result_path = os.path.join(os.getcwd(), "data_path", 'result') #model_recognize预测结果
model_classify_path = os.path.join(os.getcwd(), "data_path_save", "TwoLayerLstm.model")
dictionary_path = os.path.join(os.getcwd(), "data_path", "word2id.pkl")
time_step = 60

def get_cmd_output(demo_sentence):
    command = 'python main.py --mode=demo --demo_sentence='+demo_sentence
    os.system(command)
    with open(result_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


if __name__ == "__main__":
    print("please input the sentences:")
    demo_sentence = input()
    if demo_sentence != '':
        # demo_sentence = "为了老人的心理健康，提园引入北京师范大学心理专业资源，开展形式各异的认知训练等保健服务。"
        function_text = get_cmd_output(demo_sentence)
        word2id = read_dictionary(dictionary_path)
        type_text = predict(model_classify_path, demo_sentence, word2id, time_step)
        print("---------------------------------")
        print("功能：{}".format(function_text))
        print("变化：[{}]".format(type_text))
        print("---------------------------------")






