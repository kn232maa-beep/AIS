import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])

# Матриця X
n = len(x)
X = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        X[i, j] = x[i] ** j
print("Матриця X:")
print(X)

# Коефіцієнти полінома
a = np.linalg.solve(X, y)
print("\nКоефіцієнти a0..a4:")
print(np.round(a, 4))


def P(t):
    result = 0
    for k in range(len(a)):
        result += a[k] * t ** k
    return result


print("\nПеревірка у вузлах:", np.round(P(x), 4))

# Графік полінома
t = np.linspace(0.1, 0.7, 200)
plt.plot(t, P(t), color='blue', label='Поліном 4-го степеня')
plt.scatter(x, y, color='red', zorder=3, label='Вузли інтерполяції')

# Значення у проміжних точках
for point in [0.2, 0.5]:
    value = P(point)
    print(f"P({point}) = {value:.4f}")
    plt.scatter(point, value, color='green', marker='s', zorder=3)
    plt.annotate(f'P({point}) = {value:.3f}', (point, value), textcoords='offset points', xytext=(8, 8))

plt.xlabel('x')
plt.ylabel('y')
plt.title('Інтерполяція поліномом 4-го степеня')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('task3_interpolation.png', dpi=150)
plt.show()
