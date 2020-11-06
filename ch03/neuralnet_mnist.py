# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


cnt = 0
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    global cnt
    if (cnt==0):
        print("a1 = np.dot(x, W1) + b1")
        print("z1 = sigmoid(a1)")
        print("     x.shape : "+str(x.shape))
        print("    W1.shape : "+str(W1.shape))
        print("    b1.shape : "+str(b1.shape))
        print("    a1.shape : "+str(a1.shape))
        print("    z1.shape : "+str(z1.shape))
        print("a2 = np.dot(z1, W2) + b2")
        print("z2 = sigmoid(a2)")
        print("    z1.shape : "+str(z1.shape))
        print("    W2.shape : "+str(W2.shape))
        print("    b2.shape : "+str(b2.shape))
        print("    a2.shape : "+str(a2.shape))
        print("    z2.shape : "+str(z2.shape))
        print("a3 = np.dot(z2, W3) + b3")
        print(" y = softmax(a3)")
        print("    z2.shape : "+str(z2.shape))
        print("    W3.shape : "+str(W3.shape))
        print("    b3.shape : "+str(b3.shape))
        print("    a3.shape : "+str(a3.shape))
        print("     y.shape : "+str(y.shape))
    cnt = cnt + 1

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
