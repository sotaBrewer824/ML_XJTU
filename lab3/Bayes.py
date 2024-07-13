import copy
import os
import nltk
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# 种类
categories = {'World': 0, 'Sci/Tech': 1, 'Sports': 2, 'Business': 3}
# 还原方法
types_word = ['stemmer', 'lemmatizer']
# 训练方法
types_train = ['TF', 'Bernoulli']
# 停用词
stopwords = set(nltk.corpus.stopwords.words('english'))
# 词干提取/词形还原
stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()


def preprocess(sent, type_word):
    """
    将输入的句子转化为单词词组,并统一为小写、去标点、去停用词、去数字、还原
    """
    # 统一为小写
    sent = sent.lower()
    # 去标点
    remove = str.maketrans('', '', string.punctuation)
    sent = sent.translate(remove)
    # 转化为单词词组
    words = nltk.word_tokenize(sent)
    # 去停用词
    words = [w for w in words if not (w in stopwords)]
    # 去数字
    words = [w for w in words if not w.isdigit()]
    # 还原:词干提取/词形还原
    if type_word == 'stemmer':
        words = [stemmer.stem(w) for w in words]

    elif type_word == 'lemmatizer':
        words = [lemmatizer.lemmatize(w) for w in words]

    return words


def load(path, type_word):
    """
    path:数据集路径
    根据指定路径读取训练集或测试集
    """
    data_x, data_y = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
        length = len(lines)
        for i, line in enumerate(lines):
            tmp = line.split('|')
            data_x.append(preprocess(tmp[1].strip(), type_word))
            data_y.append(tmp[0])
            if i % 1000 == 0:
                print('loading:{}/{}'.format(i, length))

    return data_x, data_y


def words2dic(train_x):
    """
    将训练集中的单词转化为词2id(从0开始)的字典
    """
    dictionary = {}
    i = 0
    for words in train_x:
        for word in words:
            if not word in dictionary:
                dictionary[word] = i
                i += 1
    return dictionary


def train_TF(train_x, train_y):
    # 词汇表
    dictionary = words2dic(train_x)

    # n(w_i in w_c) 词频-种类 矩阵(维度:词汇数x类别数)
    words_frequency = np.zeros((len(dictionary), len(categories)), dtype=int)

    # n(c,text) 每类下的句总数(维度:类别数x1)
    category_sents = np.zeros(len(categories), dtype=int)

    for words, cate in zip(train_x, train_y):
        for word in set(words):
            words_frequency[dictionary[word]][categories[cate]] += 1
        category_sents[categories[cate]] += 1

    # p(c) (维度:类别数x1)
    p_c = category_sents / len(train_y)

    # n(w_c) 每类下的词总数(维度:类别数x1)
    category_words = np.sum(words_frequency, 0)

    # p(w_i|c) (维度:词汇数x类别数)
    p_stat = (words_frequency + 1) / (category_words + len(dictionary))

    return p_stat, dictionary, p_c


def train_Bernoulli(train_x, train_y):
    # 词汇表
    dictionary = words2dic(train_x)

    # n(w_i in w_c) 词频-种类 矩阵(维度:词汇文档数x类别数)
    words_frequency = np.zeros((len(dictionary), len(categories)), dtype=int)

    # n(c,text) 每类下的句总数(维度:类别数x1)
    category_sents = np.zeros(len(categories), dtype=int)

    for word in dictionary:
        for words, cate in zip(train_x, train_y):
            if word in words:
                words_frequency[dictionary[word]][categories[cate]] += 1
            category_sents[categories[cate]] += 1

    # p(c) (维度:类别数x1)
    p_c = category_sents / len(train_y)

    # p(w_i|c) (维度:词汇数x类别数)
    p_stat = (words_frequency + 1) / (category_sents + 2)

    return p_stat, dictionary, p_c


def test(data_x, data_y, p_stat, dictionary, p_c, type_train):
    """
    批量数据测试,计算准确率
    """
    # 统计预测正确的数目
    count = 0
    real = np.zeros(len(data_y))
    word_vec = np.zeros((len(dictionary), len(data_y)))
    # 计算argmax(...)
    if type_train == 'TF':
        for i, (words, cate) in enumerate(zip(data_x, data_y)):
            for word in words:
                if word in dictionary:
                    word_vec[dictionary[word]][i] += 1
            real[i] = categories[cate]
        res = np.dot(np.transpose(word_vec), np.log(p_stat)) + np.log(p_c)
        count = len(data_y) - np.count_nonzero(real - np.argmax(res, axis=1))
    elif type_train == 'Bernoulli':
        for i, (words, cate) in enumerate(zip(data_x, data_y)):
            for word in dictionary:
                if word in words:
                    word_vec[dictionary[word]][i] = 1
            real[i] = categories[cate]
        res = np.dot(np.transpose(word_vec), np.log(
            p_stat)) + np.dot(1 - np.transpose(word_vec), np.log(1 - p_stat)) + np.log(p_c)
        count = len(data_y) - np.count_nonzero(real - np.argmax(res, axis=1))

    print('Accuracy: {}/{} {}%'.format(count,
          len(data_y), round(100*count/len(data_y), 2)))


if __name__ == '__main__':
    p_stat, dictionary, p_c = [], [], []

    for type_word in types_word:
        train_x, train_y = load(
            os.getcwd()+'\\data\\news_category_train_mini.csv', type_word)
        test_x, test_y = load(
            os.getcwd() + '\\data\\news_category_test_mini.csv', type_word)
        for type_train in types_train:

            if type_train == 'TF':
                p_stat, dictionary, p_c = train_TF(train_x, train_y)
            elif type_train == 'Bernoulli':
                p_stat, dictionary, p_c = train_Bernoulli(train_x, train_y)
            print(type_word, type_train)
            # 训练集上的准确率
            test(train_x, train_y, p_stat, dictionary, p_c, type_train)
            # 测试集上的准确率
            test(test_x, test_y, p_stat, dictionary, p_c, type_train)
