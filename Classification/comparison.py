
import numpy as np
import matplotlib.pyplot as plt
import models

POINTS = 10000

def draw_points(m: int):
    """
    return m points according to the questions distribution
    """
    X = np.random.multivariate_normal([0, 0], np.identity(2), m)
    f_arr = np.array([0.3, -0.5])
    y = np.sign(X @ f_arr.T + 0.1)

    return X, y

if __name__ == '__main__':
    perceptron = models.Perceptron()
    svm = models.SVM()
    lda = models.LDA()

    test_X, test_y = draw_points(POINTS)
    pre_accuracy = np.zeros(5)
    svm_accuracy = np.zeros(5)
    lda_accuracy = np.zeros(5)
    samples = [5, 10, 15, 25, 70]
    repeat = 250

    for i, m in enumerate(samples):
        # Question 9
        X, y = draw_points(m)
        while (y == 1).all() or (y == -1).all():
            X, y = draw_points(m)
        colors = np.where(y == 1, 'blue', 'orange')

        perceptron.fit(X, y)
        svm.fit(X, y)

        plt.plot([-3, 3], [-3 * 0.3 / 0.5 + 0.1 / 0.5, 3 * 0.3 / 0.5 + 0.1 / 0.5],
                    label='f')

        plt.plot([-3, 3], [
            -3 * perceptron.model[1] / -perceptron.model[2] + perceptron.model[0] / -perceptron.model[2],
            3 * perceptron.model[1] / -perceptron.model[2] + perceptron.model[0] / -perceptron.model[2]],
             label='perceptron')

        plt.plot([-3, 3], [
            -3 * svm.model.coef_[0, 0] / -svm.model.coef_[0, 1] + svm.model.intercept_ / -svm.model.coef_[0, 1],
            3 * svm.model.coef_[0, 0] / -svm.model.coef_[0, 1] + svm.model.intercept_ / -svm.model.coef_[0, 1]], label='SVM')

        plt.scatter(X[:, 0], X[:, 1], color=colors)
        plt.legend()
        plt.title(f'Question 9 - m = {m}')
        plt.savefig(f"Question 9 graph {i}")
        plt.show()


# question 10
        for j in range(repeat):
            X, y = draw_points(m)
            while (y == 1).all() or (y == -1).all():
                X, y = draw_points(m)

            perceptron.fit(X, y)
            svm.fit(X, y)
            lda.fit(X, y)

            pre_accuracy[i] += perceptron.score(test_X, test_y)['accuracy']/repeat
            svm_accuracy[i] += svm.score(test_X, test_y)['accuracy']/repeat
            lda_accuracy[i] += lda.score(test_X, test_y)['accuracy']/repeat
    plt.plot(samples, pre_accuracy, label='Perceptron')
    plt.plot(samples, svm_accuracy, label='SVM')
    plt.plot(samples, lda_accuracy, label='LDA')
    plt.legend()
    plt.title("Question 10")
    plt.savefig("Question 10 graph")
    plt.show()





