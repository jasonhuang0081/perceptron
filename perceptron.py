import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.
thres = 0.001
from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=-1):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.epoch = deterministic

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.row, self.col = X.shape
        if initial_weights is None:
            self.weights = self.initialize_weights()
        else:
            self.weights = initial_weights
        # self.weights = self.initialize_weights() if not initial_weights else initial_weights
        # a, b = self.weights.shape
        # if a == 1:
        #     self.weights = np.transpose(self.weights)
        aug = np.ones((self.row, 1))
        X = np.concatenate((X,aug),axis=1)
        self.weights = self.weights.reshape(-1,1)
        self.epoch_error = []

        if self.epoch == -1:
            self.num_epoch = 0
            preAccuracy = 0
            while True:
                if self.shuffle is True:
                    X, y = self._shuffle_data(X, y)
                accuracy = self.iterate(X,y)
                self.epoch_error.append(1 - accuracy)
                self.num_epoch += 1
                if abs(accuracy - preAccuracy) < thres:
                    break
                preAccuracy = accuracy
        else:
            self.num_epoch = self.epoch
            for i in range(self.epoch):
                if self.shuffle is True:
                    X, y = self._shuffle_data(X, y)
                self.iterate(X,y)

        return self

    def iterate(self, X, y):
        for i in range(self.row):
            net = np.dot(X[i,:],self.weights)
            output = 0
            if net[0] > 0:
                output = 1
            if output != y[i]:
                dw = self.lr*(y[i, 0] - output)*X[i, :]
                self.weights = self.weights + np.reshape(dw,(self.col + 1,1))
        X = np.delete(X, np.s_[-1:], axis=1)
        accuracy = self.score(X, y)
        return accuracy

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        row,col = X.shape
        aug = np.ones((row, 1))
        X = np.concatenate((X,aug),axis=1)
        result = np.zeros(row)
        for i in range(row):
            net = np.dot(X[i, :], self.weights)
            if net[0] > 0:
                result[i] = 1
            else:
                result[i] = 0
        result = np.reshape(result, (-1, 1))
        return result, result.shape

    def predict_net(self, X):           # used for more multiple perceptron to get net values
        row, col = X.shape
        aug = np.ones((row, 1))
        X = np.concatenate((X, aug), axis=1)
        nets = np.zeros(row)
        for i in range(row):
            nets[i] = np.dot(X[i, :], self.weights)
        nets = np.reshape(nets, (-1, 1))
        return nets

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        weight = np.random.rand(self.col + 1, 1)
        # weight = np.zeros((self.col + 1,1))
        return weight

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:output
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        output, shape = self.predict(X)
        row, col = output.shape
        correct = 0
        for i in range(row):
            if output[i, 0] == y[i, 0]:
                correct += 1
        return correct/row

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        rand = np.random.permutation(self.row)
        X = X[rand]
        y = y[rand]
        return X, y

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
