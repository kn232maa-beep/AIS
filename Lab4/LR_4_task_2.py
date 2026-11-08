import numpy as np
import matplotlib.pyplot as plt

# Варіант 14
X = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0])
Y = np.array([-5, -4, 0.7, 1.25, 3, 5])

# Метод найменших квадратів
A = np.column_stack((np.ones(len(X)), X))
beta = np.linalg.solve(A.T @ A, A.T @ Y)
Y_pred = A @ beta

S = np.sum((Y - Y_pred) ** 2)
r2 = 1 - S / np.sum((Y - Y.mean()) ** 2)

print(f"b0 = {beta[0]:.4f}")
print(f"b1 = {beta[1]:.4f}")
print(f"Апроксимуюча функція: y = {beta[0]:.4f} + ({beta[1]:.4f})x")
print(f"Сума квадратів похибок S = {S:.4f}")
print(f"R2 = {r2:.4f}")

print("\n   X       Y     Y розрах.")
for a, b, c in zip(X, Y, Y_pred):
    print(f"{a:5.2f} {b:7.2f} {c:9.3f}")

# Графік
line_x = np.linspace(X.min() - 0.3, X.max() + 0.3, 100)
line_y = beta[0] + beta[1] * line_x

plt.scatter(X, Y, color='red', zorder=3, label='Експериментальні точки')
plt.plot(line_x, line_y, color='blue', label=f'y = {beta[0]:.2f} + ({beta[1]:.2f})x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Метод найменших квадратів, варіант 14')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('task2_least_squares.png', dpi=150)
plt.show()
