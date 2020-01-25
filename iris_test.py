from sklearn.linear_model import Perceptron
import numpy as np
from arff import Arff
from perceptron import PerceptronClassifier


def split(X, y):            # function to split 70/30
    row, col = X.shape
    rand = np.random.permutation(row)
    X = X[rand]
    y = y[rand]

    num_eval = round(row * 0.3)
    X_eval = X[0:num_eval, :]
    X_train = X[num_eval:, :]
    y_eval = y[0:num_eval, :]
    y_train = y[num_eval:, :]
    return X_train, y_train, X_eval, y_eval

def process_label(y_in, class_keep):        # modify labels so each perceptron can use
    y = np.copy(y_in)
    rows = np.where(y[:,0] == class_keep)
    y[rows, :] = -1
    rows2 = np.where(y[:,0] != class_keep)
    y[rows2, :] = 0
    y[rows, :] = 1
    return y

def evaluate_accuracy(P1, P2, P3, y, X_eval):   # used to evaluate final accuracy
    out1, size = P1.predict(X_eval)
    net1 = P1.predict_net(X_eval)
    out2, size = P2.predict(X_eval)
    net2 = P2.predict_net(X_eval)
    out3, size = P3.predict(X_eval)
    net3 = P3.predict_net(X_eval)
    row, col = y.shape
    correct = 0
    for i in range(row):
        if y[i,0] == 0:
            if out1[i,0] == 1 and out2[i,0] == 0 and out3[i,0] == 0:
                correct += 1
            elif out1[i,0] == 1 and net1[i,0] > net2[i,0] and net1[i,0] > net3[i,0]:
                correct += 1
        elif y[i,0] == 1:
            if out2[i,0] == 1 and out1[i,0] == 0 and out3[i,0] == 0:
                correct += 1
            elif out2[i,0] == 1 and net2[i,0] > net1[i,0] and net2[i,0] > net3[i,0]:
                correct += 1
        elif y[i,0] == 2:
            if out3[i,0] == 1 and out1[i,0] == 0 and out2[i,0] == 0:
                correct += 1
            elif out3[i,0] == 1 and net3[i,0] > net1[i,0] and net3[i,0] > net2[i,0]:
                correct += 1
    return correct / row

if __name__ == "__main__":
    mat = Arff("iris.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    X_train, y_train, X_eval, y_eval = split(data, labels)

    P1 = PerceptronClassifier(lr=0.1, shuffle=True)     # train each perceptron
    y = process_label(y_train, 0)
    row, col = X_train.shape
    initial_weight = np.zeros((col + 1,1))
    P1.fit(X_train, y, initial_weight)
    print("P1 training accuracy = ", P1.score(X_train, y))

    P2 = PerceptronClassifier(lr=0.1, shuffle=True)
    y = process_label(y_train, 1)
    row, col = X_train.shape
    initial_weight = np.zeros((col + 1,1))
    P2.fit(X_train, y, initial_weight)
    print("P2 training accuracy = ", P2.score(X_train, y))

    P3 = PerceptronClassifier(lr=0.1, shuffle=True)
    y = process_label(y_train, 2)
    row, col = X_train.shape
    initial_weight = np.zeros((col + 1,1))
    P3.fit(X_train, y, initial_weight)
    print("P3 training accuracy = ", P3.score(X_train, y))

    result = evaluate_accuracy(P1, P2, P3, y_eval, X_eval)

    print("total accuracy on testing set = ", result)
