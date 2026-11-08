import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Дані з попереднього завдання (варіант 4)
np.random.seed(14)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)


def plot_learning_curves(model, X, y, title, filename):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=14)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.figure()
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.ylim(0, 6)
    plt.xlabel('Розмір навчального набору')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=150)
    print(f"{title}: RMSE train = {np.sqrt(train_errors[-1]):.3f}, RMSE val = {np.sqrt(val_errors[-1]):.3f}")


# Лінійна модель
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, 'Лінійна модель', 'task2_6_linear.png')

# Поліном 10-го степеня
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression, X, y, 'Поліном 10-го степеня', 'task2_6_poly10.png')

# Поліном 2-го степеня
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y, 'Поліном 2-го степеня', 'task2_6_poly2.png')

plt.show()
