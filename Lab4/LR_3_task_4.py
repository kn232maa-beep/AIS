import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Лінійна регресія
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest)

print("Коефіцієнти регресії (coef_):")
print(np.round(regr.coef_, 2))
print("Вільний член (intercept_) =", round(regr.intercept_, 2))
print("R2 =", round(r2_score(ytest, ypred), 2))
print("MAE =", round(mean_absolute_error(ytest, ypred), 2))
print("MSE =", round(mean_squared_error(ytest, ypred), 2))

# Графік
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.savefig('task2_4_diabetes.png', dpi=150)
plt.show()
