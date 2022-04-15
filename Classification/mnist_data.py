import models
import numpy as np
from time import time
import matplotlib.pyplot as plt


def draw_points(m: int):
    """
    return m points according to the questions distribution
    """
    m_range = np.arange(y_train.size)
    np.random.shuffle(m_range)
    rand_m = m_range[:m]
    x_m, y_m = x_train[rand_m], y_train[rand_m]
    while (y_m == 1).all() or (y_m == -1).all():
        np.random.shuffle(m_range)
        rand_m = m_range[:m]
        x_m, y_m = x_train[rand_m], y_train[rand_m]
    return x_m, y_m


def rearrange_data(X: np.array):
    """
    reshape the data to be in the shape: (n, 784)
    """
    return np.reshape(X, X.shape[0], 7884)


if __name__ == '__main__':

    image_size = 28

    train_data = np.loadtxt("mnist_train.csv", delimiter=",")
    test_data = np.loadtxt("mnist_test.csv", delimiter=",")

    y_train, x_train = np.asfarray(train_data[:, 0]), np.asfarray(train_data[:, 1:])
    y_test, x_test = np.asfarray(test_data[:, 0]), np.asfarray(test_data[:, 1:])

    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

# Question 12 play time

    # plt.imshow(x_test[70].reshape(image_size, image_size))
    # plt.show()
    # plt.imshow(x_test[2].reshape(image_size, image_size))
    # plt.show()
    # plt.imshow(x_test[3].reshape(image_size, image_size))
    # plt.show()

    samples = [50, 100, 300, 500]

    repeat = 50

    log_reg = models.Logistic()
    soft_svm = models.SoftSVM()
    tree = models.DecisionTree(3)
    k_n_n = models.KNearestNeighbors()

    log_reg_accuracy, log_reg_time = np.zeros(4), np.zeros(4)
    soft_svm_accuracy, soft_svm_time = np.zeros(4), np.zeros(4)
    tree_accuracy, tree_time = np.zeros(4), np.zeros(4)
    k_n_n_accuracy, k_n_n_time = np.zeros(4), np.zeros(4)

    # Question 14
    for j in range(repeat):
        for i, m in enumerate(samples):
            X, y = draw_points(m)

            prev_time = time()
            log_reg.model.fit(X, y)
            log_reg_accuracy[i] += log_reg.score(x_test, y_test)['accuracy'] / repeat
            log_reg_time[i] += (time() - prev_time) / repeat

            prev_time = time()
            soft_svm.model.fit(X, y)
            soft_svm_accuracy[i] += soft_svm.score(x_test, y_test)['accuracy'] / repeat
            soft_svm_time[i] += (time() - prev_time) / repeat

            prev_time = time()
            tree.model.fit(X, y)
            tree_accuracy[i] += tree.score(x_test, y_test)['accuracy'] / repeat
            tree_time[i] += (time() - prev_time) / repeat

            prev_time = time()
            k_n_n.model.fit(X, y)
            k_n_n_accuracy[i] += k_n_n.score(x_test, y_test)['accuracy'] / repeat
            k_n_n_time[i] += (time() - prev_time) / repeat

    print(f'Logistic Regression: {log_reg_time}\n '
          f'Soft SVM: {soft_svm_time}\n'
          f'Decision Tree: {tree_time}\n'
          f'K-Nearest Neighbors: {k_n_n_time}')

    plt.plot(samples, log_reg_accuracy, label='Logistic Regression')
    plt.plot(samples, soft_svm_accuracy, label='Soft SVM')
    plt.plot(samples, tree_accuracy, label='Decision Tree')
    plt.plot(samples, k_n_n_accuracy, label='K-Nearest Neighbors')
    plt.title('Question 14')
    plt.legend()
    plt.savefig('Question 14')
    plt.show()




