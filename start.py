from perceptron import PerceptronClassifier
from arff import Arff
from graph_tools import *
import pandas as pd
import matplotlib.pyplot as plt

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

# this main function has many code commented out but still keep it here to speed up using perceptron and
# its settings
if __name__ == "__main__":
    # mat = Arff("linsep2nonorigin.arff", label_count=1)
    # mat = Arff("data_banknote_authentication.arff", label_count=1)
    mat = Arff("votingMissingValuesReplaced.arff", label_count=1)
    # mat = Arff("test2.arff", label_count=1)

    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)

    X_train, y_train, X_eval, y_eval = split(data, labels)  # split data in 70/30


    # PClass = PerceptronClassifier(lr=0.1, shuffle=False, deterministic=10)    #initialize perceptron with settings
    # PClass = PerceptronClassifier(lr=0.1, shuffle=False)
    PClass = PerceptronClassifier(lr=0.1, shuffle=True)

    row, col = data.shape               # using all zeros initial weight, if not provided, it will do random
    initial_weight = np.zeros((col + 1,1))
    PClass.fit(X_train, y_train, initial_weight)        # train the perceptron
    # PClass.fit(data, labels, initial_weight)

    # graph(data[:,0].reshape(-1,1),data[:,1].reshape(-1,1),labels, "linearly separable")    # plotting scatter plot
    # index = list(range(1, PClass.num_epoch + 1))      # plotting scatter plot with each epoch's error during training
    # plt.scatter(index, PClass.epoch_error)
    # plt.plot(index, PClass.epoch_error)
    # plt.show()
    # y = lambda x: -1.3333* x              # plotting the line
    # y = lambda x: 0.8181 * x
    # graph_function(y)

    Accuracy = PClass.score(X_eval, y_eval)
    # Accuracy = PClass.score(data, labels)
    print("Accuracy = [{:.2f}]".format(Accuracy))
    weights = PClass.get_weights()
    np.set_printoptions(precision=4)
    print("Final Weights = ", weights)
    print("epochs = ", PClass.num_epoch)

    # df = pd.DataFrame(PClass.epoch_error)             # write to excel codes
    # filepath = '1.csv'
    # df.to_csv(filepath, index=True)

