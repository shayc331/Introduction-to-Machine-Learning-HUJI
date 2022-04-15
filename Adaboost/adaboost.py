"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np
import ex4_tools as tools
import matplotlib.pyplot as plt

T = 500
TEST_AMOUNT = 200

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = np.array([None]*T)     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        D = np.array([1/len(y)] * len(y))
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            epsilon_t = np.sum(0.5 * D * np.abs(y - self.h[t].predict(X)))
            self.w[t] = 0.5 * np.log((1 / epsilon_t) - 1)
            D = (D * np.exp(-y * self.w[t] * self.h[t].predict(X))) / np.sum(D * np.exp(-y * self.w[t] * self.h[t].predict(X)))
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        return np.sign([sum(self.h[t].predict(X) * self.w[t] for t in range(max_t))])



    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        result = y_hat[y_hat != y]
        return len(result) / len(y)


def question_13(z):
    train_error = [adaboost.error(X_train, y_train, t) for t in range(1, T + 1)]
    test_error = [adaboost.error(X_test, y_test, t) for t in range(1, T + 1)]
    plt.plot(np.arange(1, T + 1), train_error, label="Train Error")
    plt.plot(np.arange(1, T + 1), test_error, label="Test Error")
    plt.legend()
    plt.title(f"QUESTION 13\nnoise = {z}")
    plt.xlabel("T")
    plt.ylabel("Error rate")
    plt.savefig(f"question 13 graph noise = {z}.png")
    plt.show()

def question_14(z):
    for i, t in enumerate(T_arr, 1):
        plt.subplot(2, 3, i)
        tools.decision_boundaries(adaboost, X_test, y_test, t)
    plt.savefig(f"question 14 graph noise = {z}.png")
    plt.show()


def question_15(z):
    min_err = 1
    T_hat = None
    for i, t in enumerate(T_range, 1):
        temp_err = adaboost.error(X_test, y_test, t)
        if temp_err < min_err:
            min_err = temp_err
            T_hat = t
    tools.decision_boundaries(adaboost, X_train, y_train, T_hat)
    plt.title(f"T_hat = {T_hat}\ntest error = {min_err}\nnoise = {z}")
    plt.savefig(f"question 15 graph noise = {z}.png")
    plt.show()


def question_16(z):
    D = adaboost.train(X_train, y_train)
    D = D / np.max(D) * 10
    tools.decision_boundaries(adaboost, X_train, y_train, T, D)
    plt.title(f"QUESTION 16\n noise = {z}")
    plt.savefig(f"question 16 noise = {z}.png")
    plt.show()

if __name__ == '__main__':

    T_arr = {5, 10, 50, 100, 200, 500}
    T_range = np.arange(1, T)
    adaboost = AdaBoost(tools.DecisionStump, T)

    for noise in [0, 0.01, 0.4]:

        X_train, y_train = tools.generate_data(5000, noise)
        adaboost.train(X_train, y_train)
        X_test, y_test = tools.generate_data(TEST_AMOUNT, noise)

        question_13(noise)
        question_14(noise)
        question_15(noise)
        question_16(noise)

