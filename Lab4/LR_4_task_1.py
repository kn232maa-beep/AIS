import numpy as np

# Експериментальні точки
x = np.array([1, 2, 3, 4])
y = np.array([6, 5, 7, 10])

# Нормальні рівняння
A = np.column_stack((np.ones(len(x)), x))
beta = np.linalg.solve(A.T @ A, A.T @ y)

residuals = y - A @ beta
S = np.sum(residuals ** 2)

print(f"b0 = {beta[0]:.2f}, b1 = {beta[1]:.2f}")
print(f"y = {beta[0]:.2f} + {beta[1]:.2f}x")
print("Похибки:", np.round(residuals, 2))
print(f"S(b0, b1) = {S:.2f}")
