import os

import numpy as np
import matplotlib.pyplot as plt

from fuzzy_mamdani import (MamdaniFIS, Rule, Variable, plot_memberships, plot_rule_viewer,
                           plot_surface, uniform_gaussmf, uniform_trimf)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(BASE_DIR, name)


#  Графік цільової функції
n = 15
x1 = np.linspace(0, 4, n)
x2 = np.linspace(0, 4, n)
y = np.zeros((n, n))
for j in range(n):
    y[j, :] = (x1 ** 2 - 8) * np.cos(x2[j])

plot_surface(x1, x2, y, ('x1', 'x2', 'y'), 'Target', out('example_1_target.png'))

#  Кроки 1-14: змінні та функції належності
FIVE = ['L', 'LA', 'A', 'HA', 'H']
FIVE_LABELS = ['L (Низький)', 'LA (Нижче середнього)', 'A (Середній)', 'HA (Вище середнього)', 'H (Високий)']

# x1: діапазон [0, 4], 3 терми з трикутними функціями належності
var_x1 = Variable('x1', (0, 4), uniform_trimf((0, 4), ['L', 'A', 'H'], ['L (Низький)', 'A (Середній)', 'H (Високий)']))
# x2: діапазон [0, 4], 5 термів із гаусовими функціями належності
var_x2 = Variable('x2', (0, 4), uniform_gaussmf((0, 4), FIVE, FIVE_LABELS))
# y: діапазон [-10, 10], 5 термів із трикутними функціями належності
var_y = Variable('y', (-10, 10), uniform_trimf((-10, 10), FIVE, FIVE_LABELS))

#  Крок 16: база знань
rules = [
    Rule(['L', 'L'], ['L']),    # 1. x1=Низький і x2=Низький -> y=Низький
    Rule(['L', 'H'], ['A']),    # 2. x1=Низький і x2=Високий -> y=Середній
    Rule(['L', 'HA'], ['H']),   # 3. x1=Низький і x2=Вище середнього -> y=Високий
    Rule(['H', 'L'], ['HA']),   # 4. x1=Високий і x2=Низький -> y=Вище середнього
    Rule(['H', 'H'], ['L']),    # 5. x1=Високий і x2=Високий -> y=Низький
    Rule(['A', 'A'], ['A']),    # 6. x1=Середній і x2=Середній -> y=Середній
    Rule(['A', 'HA'], ['HA']),  # 7. x1=Середній і x2=Вище середнього -> y=Вище середнього
    Rule(['L', 'LA'], ['LA']),  # 8. x1=Низький і x2=Нижче середнього -> y=Нижче середнього
    Rule(['A', 'HA'], ['A']),   # 9. x1=Середній і x2=Вище середнього -> y=Середній
]

fis = MamdaniFIS('first', [var_x1, var_x2], [var_y], rules)
fis.print_structure()
print()
fis.print_rules()
fis.write_fis(out('first.fis'))
plot_memberships([var_x1, var_x2, var_y], out('example_2_memberships.png'))

#  Візуалізація виводу та поверхня «входи-вихід»
point = [2, 2]
result = fis.evaluate(point)[0]
print(f"\nНечіткий вивід для x1 = {point[0]}, x2 = {point[1]}: y = {result:.4f}")
print(f"Точне значення функції: y = {(point[0] ** 2 - 8) * np.cos(point[1]):.4f}")
plot_rule_viewer(fis, point, out('example_3_rule_viewer.png'))

sx1, sx2, sy = fis.surface(n=15)
plot_surface(sx1, sx2, sy, ('x1', 'x2', 'y'), 'Поверхня «входи-вихід» СНВ', out('example_4_fis_surface.png'))

#  Порівняння СНВ із цільовою функцією
error = sy - y
print("\nПорівняння поверхні СНВ із цільовою функцією на сітці 15 x 15:")
print(f"Середньоквадратична похибка (RMSE): {np.sqrt(np.mean(error ** 2)):.4f}")
print(f"Середня абсолютна похибка (MAE):   {np.mean(np.abs(error)):.4f}")
print(f"Максимальна абсолютна похибка:     {np.max(np.abs(error)):.4f}")
print(f"Діапазон цільової функції: [{y.min():.3f}, {y.max():.3f}]; діапазон СНВ: [{sy.min():.3f}, {sy.max():.3f}]")

plt.show()
