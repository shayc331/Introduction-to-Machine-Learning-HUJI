import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_regression(X, y):
    singular_values = np.linalg.svd(X, compute_uv=False)
    X_dagger = np.linalg.pinv(X)
    w_hat = X_dagger @ y
    return w_hat, singular_values


def predict(X, w):
    return X @ w


def mse(response, prediction):
    return (np.square(response - prediction)).mean()


def load_data(path):
    df = pd.read_csv(path).drop_duplicates().dropna()
    df.drop(columns=['id', 'sqft_living'], inplace=True)
    for feature in {'price', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'yr_built'}:
        df = df[df[feature] > 0]
    df = pd.get_dummies(df, prefix='_zipcode', columns=['zipcode'])
    df['date'] = df['date'].map(lambda x: x[:3] + '0')
    df = pd.get_dummies(df, prefix='_sale decade', columns=['date'])

    df.insert(0, 'built or renovated', np.maximum(df['yr_built'], df['yr_renovated']))
    df.drop(columns=['yr_built', 'yr_renovated'], inplace=True)

    df.insert(0, '_intercept', 1)
    return df


def plot_singular_values(singular_values):
    y = sorted(singular_values, reverse=True)
    plt.plot(y, color='purple')
    plt.title('singular values')
    plt.xlabel("number of singular values")
    plt.ylabel("singular values")
    plt.savefig("singular values")
    plt.show()


def q15(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    plot_singular_values(singular_values)


def split_data(df):
    test_set = df.sample(frac=.25)
    train_set = df.drop(test_set.index)
    y_test = test_set.get('price')
    y_train = train_set.get('price')
    train_set.drop(columns='price', inplace=True)
    test_set.drop(columns='price', inplace=True)
    return test_set, train_set, y_test, y_train

def q16(df):
    test_set, train_set, y_test, y_train = split_data(df)
    length = len(y_train)
    fit_func = lambda x: fit_linear_regression(train_set[:int(x * length)], y_train[:int(x * length)])
    w_arr = [fit_func(p)[0] for p in np.arange(0.01, 1.01, 0.01)]
    predict_arr = [predict(test_set, w) for w in w_arr]
    plt.plot(np.arange(1, 101), np.log([mse(y_test, i) for i in predict_arr]), color='green')
    plt.title("MSE ACCORDING TO SIZE OF THE TRAIN SET")
    plt.xlabel("percentage of the train set")
    plt.ylabel("LOG MSE")
    plt.savefig("MSE graph")
    plt.show()


def feature_evaluation(X, y):
    for feature, val in X.iteritems():
        if feature[0] != '_':  # not categorical feature
            corr = np.cov(val, y)[0, 1] / np.sqrt(np.var(val) * np.var(y))
            plt.scatter(val, y, color='red')
            plt.xlabel(feature)
            plt.ylabel('price')
            plt.title(f'{feature} and price. corr = {corr}')
            plt.savefig(f'{feature} evaluation graph')
            plt.show()


if __name__ == '__main__':
    data = load_data("kc_house_data.csv")
    q15(data)
    q16(data)
    response = data['price']
    data.drop(columns=['price'], inplace=True)
    feature_evaluation(data, response)




