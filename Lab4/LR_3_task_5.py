import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Варіант 4
np.random.seed(14)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)

plt.figure()
plt.scatter(X, y, s=15)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Згенеровані дані')
plt.savefig('task2_5_data.png', dpi=150)

# Лінійна регресія
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

print("Лінійна регресія:")
print(f"y = {linear.coef_[0][0]:.2f}x + {linear.intercept_[0]:.2f}")

order = X[:, 0].argsort()

plt.figure()
plt.scatter(X, y, s=15)
plt.plot(X[order], y_linear[order], color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія')
plt.savefig('task2_5_linear.png', dpi=150)

# Поліноміальна регресія 2-го степеня
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("\nX[0] =", X[0])
print("X_poly[0] =", X_poly[0])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
y_poly = lin_reg.predict(X_poly)

print("\nintercept_ =", lin_reg.intercept_)
print("coef_ =", lin_reg.coef_)

print("\nМодель варіанта: y = 0.7x^2 + 1.0x + 3 + гаусів шум")
print(f"Отримана модель: y = {lin_reg.coef_[0][1]:.2f}x^2 + {lin_reg.coef_[0][0]:.2f}x + {lin_reg.intercept_[0]:.2f}")

plt.figure()
plt.scatter(X, y, s=15)
plt.plot(X[order], y_poly[order], color='red', linewidth=2, label='Прогноз')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Поліноміальна регресія 2-го степеня')
plt.legend()
plt.savefig('task2_5_polynomial.png', dpi=150)

# Оцінка якості
print("\n            R2      MAE     MSE")
for name, pred in [('Лінійна', y_linear), ('Поліном.', y_poly)]:
    print(f"{name:<9} {r2_score(y, pred):6.3f} {mean_absolute_error(y, pred):7.3f} {mean_squared_error(y, pred):7.3f}")

plt.show()
