import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt'

# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Навчальний та тестовий набори
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Лінійний регресор
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)


def print_metrics(y_true, y_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_true, y_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_true, y_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_true, y_pred), 2))
    print("Explained variance score =", round(sm.explained_variance_score(y_true, y_pred), 2))
    print("R2 score =", round(sm.r2_score(y_true, y_pred), 2))


print("Linear Regressor performance:")
print_metrics(y_test, y_test_pred)

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression(tol=1e-14)
poly_linear_model.fit(X_train_transformed, y_train)

print("\nLinear regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

# Якість поліноміальної моделі на тестових даних
y_test_pred_poly = poly_linear_model.predict(polynomial.transform(X_test))
print("\nPolynomial Regressor performance:")
print_metrics(y_test, y_test_pred_poly)
