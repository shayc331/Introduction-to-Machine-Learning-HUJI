import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn import datasets

K = 5
kf = KFold(n_splits=K)


def f(x, epsilon):
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2) + epsilon

def question_4():
    # constants
    m = 1500
    TRAIN_AMOUNT = 500
    VALIDATION_AMOUNT = 1000

    TEST_AMOUNT = 500
    X = [-3.2, 2.2]
    mu = 0
    d = 15

    # question 4
    # a + b
    for sigma in [1, 5]:

        samples = np.random.uniform(X[0], X[1], m)
        noise = np.random.normal(mu, sigma, m)
        D = np.array([f(samples[i], noise[i]) for i in range(VALIDATION_AMOUNT)])
        S = np.array(D[:TRAIN_AMOUNT])
        V = np.array(D[TRAIN_AMOUNT: VALIDATION_AMOUNT])
        T = np.array([f(samples[i], noise[i]) for i in range(VALIDATION_AMOUNT, m)])

        # c
        trained_h = [np.polyfit(samples[:TRAIN_AMOUNT], S[:TRAIN_AMOUNT], i + 1) for i in range(d)]
        h_hat_index, best_mse = None, None
        for i in range(d):
            # val = np.polyval(trained_h[i], samples[TRAIN_AMOUNT: VALIDATION_AMOUNT]) - V
            # print(val)
            cur_mse = np.mean(np.square(np.polyval(trained_h[i], samples[TRAIN_AMOUNT: VALIDATION_AMOUNT]) - V))
            if best_mse is None or cur_mse < best_mse:
                best_mse = cur_mse
                h_hat_index = i

        # d

        mse_val_lst = np.zeros(d)
        mse_train_lst = np.zeros(d)
        for train_index, validation_index in kf.split(D):
            # h = np.polyfit(samples[train_index], D[train_index], 1)
            trained_h = [np.polyfit(samples[train_index], D[train_index], i + 1) for i in range(d)]
            mse_train_lst += [np.mean(np.square(np.polyval(trained_h[i], samples[train_index]) - D[train_index]))
                              for i in range(d)]

            mse_val_lst += [np.mean(np.square(np.polyval(trained_h[i], samples[validation_index]) - D[validation_index]))
                            for i in range(d)]
        mse_val_lst /= K
        mse_train_lst /= K

        # e - g
        plt.plot(np.arange(1, d + 1), mse_train_lst, label="Train MSE")
        plt.plot(np.arange(1, d + 1), mse_val_lst, label="Validation MSE")
        plt.xlabel("Degree")
        plt.ylabel("MSE")
        plt.legend()
        plt.title(f"Training and Validation errors sigma = {sigma}")

        plt.savefig(f"question 4 sigma = {sigma}")
        plt.show()

        d_star = np.argmin(mse_val_lst)
        h_star = np.polyfit(samples[:VALIDATION_AMOUNT], D, d_star)
        test_error = np.mean(np.square(np.polyval(h_star, samples[VALIDATION_AMOUNT: m]) - T))
        print(f"Test error with sigma = {sigma} is {test_error} \n")

def fit_and_predict(learner, train_index, validation_index, train_error_lst, valid_error_lst, i):
    learner.fit(training_set[train_index], train_y[train_index])
    train_error_lst[i] += np.mean(np.square(learner.predict(training_set[train_index]) - train_y[train_index])) / K

    valid_error_lst[i] += np.mean(np.square(learner.predict(training_set[validation_index]) - train_y[validation_index])) / K

def predict_model(learner):
    return np.mean(np.square(learner.predict(test_set) - test_y))


def question_5():
    ridge_train_error = np.zeros(100)
    lasso_train_error = np.zeros(100)
    ridge_valid_error = np.zeros(100)
    lasso_valid_error = np.zeros(100)
    lambda_values = np.linspace(0.01, 2, 100)
    # c
    ridge_list = [Ridge(alpha) for alpha in lambda_values]
    lasso_list = [Lasso(alpha) for alpha in lambda_values]
    for i, (ridge, lasso) in enumerate(zip(ridge_list, lasso_list)):
        for train_index, validation_index in kf.split(training_set):
            fit_and_predict(ridge, train_index, validation_index, ridge_train_error, ridge_valid_error, i)
            fit_and_predict(lasso, train_index, validation_index, lasso_train_error, lasso_valid_error, i)

    # d
    plt.plot(lambda_values, ridge_train_error, label="Ridge Train Error")
    plt.plot(lambda_values, ridge_valid_error, label="Ridge Validation Error")
    plt.plot(lambda_values, lasso_train_error, label="Lasso Train Error")
    plt.plot(lambda_values, lasso_valid_error, label="Lasso Validation Error")

    plt.xlabel("Lambda values")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Training and Validation Errors Ridge")
    plt.savefig("question 5 graph")
    plt.show()

    # e
    ridge_lambda = lambda_values[np.argmin(ridge_valid_error)]
    lasso_lambda = lambda_values[np.argmin(lasso_valid_error)]
    print(f"Ridge best lambda = {ridge_lambda}")
    print(f"Lasso best lambda = {lasso_lambda}\n")

    #f
    best_ridge = ridge_list[np.argmin(ridge_valid_error)]
    best_lasso = lasso_list[np.argmin(lasso_valid_error)]
    linear_regresion = LinearRegression()
    linear_regresion.fit(training_set, train_y)

    linear_regresion_error = predict_model(linear_regresion)
    best_ridge_error = predict_model(best_ridge)
    best_lasso_error = predict_model(best_lasso)

    print(f"Linear Regression Error = {linear_regresion_error}")
    print(f"Ridge Error = {best_ridge_error}")
    print(f"Lasso Error = {best_lasso_error}")





if __name__ == '__main__':
    question_4()
    X, y = datasets.load_diabetes(return_X_y=True)
    training_set = X[:50]
    test_set = X[50:]
    train_y = y[:50]
    test_y = y[50:]
    question_5()