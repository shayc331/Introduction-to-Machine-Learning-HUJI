
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def model_score(model, X, y):
    """
    make the prediction for the model according to X and y and return the result as a dictionary
    """
    y_len = len(y)
    result = model.predict(X)
    FN, TP, FP, TN = 0, 0, 0, 0
    for predict_val, true_val in zip(result, y):
        if predict_val > true_val:
            FP += 1
        elif predict_val < true_val:
            FN += 1
        elif predict_val == 1 and true_val == 1:
            TP += 1
        elif predict_val == -1 and true_val == -1:
            TN += 1
    return {"num samples": len(y), "errors": FP + FN, "accuracy": (TP + TN) / y_len,
            "FPR": FP / (FN + TN), "TPR": TP / (TP + FP), "precision": TP / (TP + FP),
            "specificty": TN / (FN + TN)}



class Perceptron:
    """
    Perceptron class
    """

    def __init__(self):
        self.model = None

    def fit(self, X: np.array, y):
        X = np.insert(X, 0, 1, axis=1)
        self.model = np.zeros(X.shape[1])
        while True:
            for i, sample in enumerate(X):
                if y[i] * (self.model.T @ sample) <= 0:
                    self.model += y[i] * sample
                    break
            else:
                return

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.sign(X @ self.model.T)

    def score(self, X, y):
        return model_score(self, X, y)


class LDA:
    """
    LDA class
    """
    def __init__(self):
        self.pr_y = None
        self.mean_y = np.zeros([2, 2])
        self.bias = None
        self.sigma_inverse_mu = None

    def fit(self, X: np.array, y: np.array):
        y_arr = [(y == 1), (y == -1)]
        self.pr_y = np.array([y_arr[0].mean(), y_arr[1].mean()])
        self.mean_y = np.array([X[y == 1].mean(axis=0), X[y == -1].mean(axis=0)]).T

        sigma = np.zeros((X.shape[1], X.shape[1]))
        for i, j in zip([1, -1], range(2)):
            sigma += (X[y == i] - self.mean_y[:, j]).T @ (X[y == i] - self.mean_y[:, j])
        sigma /= y.size  # len of rows
        inverse_sigma = np.linalg.inv(sigma)
        self.bias = -0.5 * np.diag(self.mean_y.T @ inverse_sigma @ self.mean_y) + np.log(self.pr_y)
        self.sigma_inverse_mu = inverse_sigma @ self.mean_y
        return

    def predict(self, X):
        return -2*(np.argmax(X @ self.sigma_inverse_mu + self.bias, axis=1)) + 1

    def score(self, X, y):
        return model_score(self, X, y)



class SVM:
    """
    SVM class
    """
    def __init__(self):
        self.model = SVC(C=1e10, kernel='linear')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return model_score(self, X, y)



class Logistic:
    """
    Logistic class
    """
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return model_score(self, X, y)


class DecisionTree:
    """
    DecisionTree class
    """
    def __init__(self, max_depth):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return model_score(self, X, y)



class SoftSVM:
    """
    SoftSVM class
    """
    def __init__(self):
        self.model = SVC()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return model_score(self, X, y)


class KNearestNeighbors:
    """
    KNearestNeighbors class
    """
    def __init__(self, neighbors=2):
        self.model = KNeighborsClassifier(n_neighbors=neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return model_score(self, X, y)

