from perceptron import PerceptronClassifier
from arff import Arff

if __name__ == "__main__":
    # mat = Arff("linsep2nonorigin.arff", label_count=1)
    mat = Arff("data_banknote_authentication.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    PClass = PerceptronClassifier(lr=0.1, shuffle=False, deterministic=10)
    PClass.fit(data, labels)
    Accuracy = PClass.score(data, labels)
    print("Accuray = [{:.2f}]".format(Accuracy))
    print("Final Weights =", PClass.get_weights())
