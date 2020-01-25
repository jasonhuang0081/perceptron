from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import numpy as np
from arff import Arff
import pandas as pd

def split(X, y):
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


if __name__ == "__main__":                  # sk-learn method
    # mat = Arff("votingMissingValuesReplaced.arff", label_count=1)
    mat = Arff("data_banknote_authentication.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    # X, y = load_digits(return_X_y=True)

    X_train, y_train, X_eval, y_eval = split(data, labels)
    y_train = y_train.flatten()
    y_eval = y_eval.flatten()
    clf = Perceptron(tol=1e-3, random_state=0, early_stopping=True, n_iter_no_change=1)
    PClass = clf.fit(X_train, y_train)
    print("weight = ", PClass.coef_)
    print("iteration = ", PClass.n_iter_)
    result = clf.score(X_eval, y_eval)
    print("accuracy = ", result)

    df = pd.DataFrame(PClass.coef_)
    filepath = '1.csv'
    df.to_csv(filepath, index=True)