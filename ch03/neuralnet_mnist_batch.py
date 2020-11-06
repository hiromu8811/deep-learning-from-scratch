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
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    global cnt
    if (cnt==0):
        print("a1 = np.dot(x, w1) + b1")
        print("z1 = sigmoid(a1)")
        print("     x.shape : "+str(x.shape))
        print("    w1.shape : "+str(w1.shape))
        print("    b1.shape : "+str(b1.shape))
        print("    a1.shape : "+str(a1.shape))
        print("    z1.shape : "+str(z1.shape))
        print("a2 = np.dot(z1, w2) + b2")
        print("z2 = sigmoid(a2)")
        print("    z1.shape : "+str(z1.shape))
        print("    w2.shape : "+str(w2.shape))
        print("    b2.shape : "+str(b2.shape))
        print("    a2.shape : "+str(a2.shape))
        print("    z2.shape : "+str(z2.shape))
        print("a3 = np.dot(z2, w3) + b3")
        print(" y = softmax(a3)")
        print("    z2.shape : "+str(z2.shape))
        print("    w3.shape : "+str(w3.shape))
        print("    b3.shape : "+str(b3.shape))
        print("    a3.shape : "+str(a3.shape))
        print("     y.shape : "+str(y.shape))
    cnt = cnt + 1

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
