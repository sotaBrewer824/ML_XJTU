import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

loss_types = ['hinge', 'exp', 'log']


def load(path, data_type):
    """
    根据指定路径读取训练集或测试集
    由于二者的数据格式略有不同，所以需要区分处理
    path:数据集路径
    data_type:"train"或"test"
    """
    data = scipy.io.loadmat(path)
    # 原始数据的label是0/1格式,需要转化为课上学的-1/1格式
    # unit8->int 0/1->-1/1
    if data_type == 'train':
        data['y'] = data['y'].astype(int) * 2 - 1
    elif data_type == 'test':
        data['ytest'] = data['ytest'].astype(int) * 2 - 1
    return data


def obj_func(train_x, train_y, W, b, lambda_, loss_type):
    """
    根据当前W和b,计算训练集样本的目标函数平均值
    """
    loss_sum = 0
    num_train = len(train_y)
    for i in range(num_train):
        resi = train_y[i] * (np.dot(np.transpose(W), train_x[i]) + b)
        resi = resi.item()
        if loss_type == 'hinge':
            loss_sum += max(0, 1 - resi)
        elif loss_type == 'exp':
            if -resi > 3:
                resi = -3
            loss_sum += np.exp(-resi)
        elif loss_type == 'log':
            if -resi > 3:
                resi = -3
            loss_sum += np.log(1 + np.exp(-resi))
    return loss_sum / num_train + lambda_ / 2 * np.linalg.norm(W, ord=2)


def plot(func_list, func_unit, loss_type, C, T, acc):
    """
    绘制模型在训练过程中的目标函数曲线
    """
    ts = [t for t in range(0, T, func_unit)]
    plt.plot(ts, func_list, 'k', label='C = {}, T = {}'.format(C, T))
    print(len(func_list))
    plt.title('{} acc={}% C={} T={}'.format(loss_type, acc, C, T))
    plt.xlabel('t')
    plt.ylabel('f(W,b)')
    if not os.path.exists(os.getcwd() + './output'):
        os.makedirs(os.getcwd() + './output')
    plt.legend()
    plt.savefig(os.getcwd(
    ) + './output/fig/{}/{}_C={}_T={}.jpg'.format(loss_type, loss_type, C, T))
    plt.cla()


def pegasos(train, test, C, T, loss_type='hinge', func_unit=100):
    """
    佩加索斯算法

    - `func_unit`:每隔func_unit次记录一次当前目标函数值,用于画图
    """
    print('C={}, T={}, loss_type:{}'.format(C, T, loss_type))
    train_x = train['X']  # 4000*1899
    train_y = train['y']  # 4000*1

    test_x = test['Xtest']  # 1000*1899
    test_y = test['ytest']  # 1000*1

    # 记录目标函数值,用于画图
    func_list = []

    # 初始化lambda_
    lambda_ = 1 / (C * len(train_y))

    # 高斯初始化权重W和偏置b
    W = np.random.randn(len(train_x[0]), 1)
    b = np.random.randn(1)

    for t in range(1, T + 1):
        eta = 1 / (lambda_ * t)
        i = np.random.randint(0, len(train_y))
        xi = train_x[i]
        yi = train_y[i]
        yxi = (yi * xi).reshape(1899, 1)
        resi = yi * (np.dot(np.transpose(W), xi) + b)
        if loss_type == 'hinge':
            if resi < 1:
                W -= eta * (lambda_ * W - yxi)
                b -= eta * (- yi)
            else:
                W -= eta * lambda_ * W
        elif loss_type == 'exp':
            if -resi > 3:
                resi = -3
            W -= eta * (lambda_ * W - yxi * np.exp(-resi))
            b -= eta * (-yi * np.exp(-resi))
        elif loss_type == 'log':
            if -resi > 3:
                resi = -3
            W -= eta * (lambda_ * W - yxi *
                        np.exp(-resi) / (1 + np.exp(-resi)))
            b -= -eta * (yi * np.exp(-resi) / (1 + np.exp(-resi)))
        # 根据当前W和b,计算训练集样本的目标函数平均值
        if t % func_unit == 0:
            func_now = obj_func(train_x, train_y, W, b, lambda_, loss_type)
            func_list.append(func_now)
            print('t = {}, func = {}'.format(t, func_now))

    # 比对test数据上预测与实际的结果,统计预测对的个数,计算准确率acc
    num_correct = 0
    for i in range(len(test_y)):
        if test_y[i] * (np.dot(np.transpose(W), test_x[i]) + b) > 0:
            num_correct += 1
    acc = 100 * num_correct / len(test_y)
    print('acc = {:.1f}%'.format(acc))
    print('func_list = {}'.format(func_list))

    return acc, func_list


if __name__ == '__main__':
    Cs = [0.05, 0.001, 0.0001]
    Ts = [5000, 10000, 100000]  # 迭代次数
    func_unit = 500  # 每隔多少次迭代计算一次目标函数
    np.random.seed(100)
    res = []
    train = load(os.getcwd() + './data/spamTrain.mat', 'train')  # 4000条
    test = load(os.getcwd() + './data/spamTest.mat', 'test')  # 1000条

    print_record = open(os.getcwd() + './output/record.txt', 'w')
    sys.stdout = print_record

    for loss_type in loss_types:
        for C in Cs:
            for T in Ts:
                acc, func_list = pegasos(
                    train, test, C, T, loss_type, func_unit)
                res.append([loss_type, C, T, acc])
                plot(func_list, func_unit, loss_type, C, T, acc)
                plt.cla()
    print_record.close()

    print_res = open(os.getcwd() + './output/res.txt', 'w')
    sys.stdout = print_res
    print(tabulate(res, headers=["LossType", "C", "T", "acc"]))
    print_res.close()