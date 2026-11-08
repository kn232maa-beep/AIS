import os

import matplotlib.pyplot as plt

from fuzzy_mamdani import MamdaniFIS, Rule, Variable, plot_memberships, plot_rule_viewer, plot_surface, uniform_trimf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(BASE_DIR, name)


#  Лінгвістичні змінні
# Діапазони в умові не задані: комфортною вважається температура 22 °C,
# швидкість зміни температури задана в °C за хвилину.
temperature = Variable('temperature', (10, 34), uniform_trimf(
    (10, 34), ['VC', 'C', 'N', 'W', 'VW'], ['Дуже холодна', 'Холодна', 'В нормі', 'Тепла', 'Дуже тепла']))
rate = Variable('rate', (-1, 1), uniform_trimf((-1, 1), ['NEG', 'ZERO', 'POS'], ['Від’ємна', 'Нуль', 'Додатня']))

# Вихід: кут повороту регулятора, градуси. Вліво (-) - режим «холод», вправо (+) - режим «тепло».
regulator = Variable('regulator', (-90, 90), uniform_trimf(
    (-90, 90), ['LB', 'LS', 'OFF', 'RS', 'RB'],
    ['Великий вліво («холод»)', 'Невеликий вліво («холод»)', 'Вимкнути',
     'Невеликий вправо («тепло»)', 'Великий вправо («тепло»)']))

# База правил
rules = [
    Rule(['VW', 'POS'], ['LB']),   # 1. Дуже тепла, додатня: «холод», великий кут вліво
    Rule(['VW', 'NEG'], ['LS']),   # 2. Дуже тепла, від'ємна: «холод», невеликий кут вліво
    Rule(['W', 'POS'], ['LB']),    # 3. Тепла, додатня: «холод», великий кут вліво
    Rule(['W', 'NEG'], ['OFF']),   # 4. Тепла, від'ємна: регулятор слід вимкнути
    Rule(['VC', 'NEG'], ['RB']),   # 5. Дуже холодна, від'ємна: «тепло», великий кут вправо
    Rule(['VC', 'POS'], ['RS']),   # 6. Дуже холодна, додатня: «тепло», невеликий кут вправо
    Rule(['C', 'NEG'], ['RB']),# 7. Холодна, від'ємна: режим «тепло». У методичці сказано «вліво», але вліво - це режим «холод»,
    # тому взято великий кут вправо.
    Rule(['C', 'POS'], ['OFF']),   # 8. Холодна, додатня: вимкнути
    Rule(['VW', 'ZERO'], ['LB']),  # 9. Дуже тепла, нуль: «холод», великий кут вліво
    Rule(['W', 'ZERO'], ['LS']),   # 10. Тепла, нуль: «холод», невеликий кут вліво
    Rule(['VC', 'ZERO'], ['RB']),  # 11. Дуже холодна, нуль: «тепло», великий кут вправо
    Rule(['C', 'ZERO'], ['RS']),   # 12. Холодна, нуль: «тепло», невеликий кут вправо
    Rule(['N', 'POS'], ['LS']),    # 13. В нормі, додатня: «холод», невеликий кут вліво
    Rule(['N', 'NEG'], ['RS']),    # 14. В нормі, від'ємна: «тепло», невеликий кут вправо
    Rule(['N', 'ZERO'], ['OFF']),  # 15. В нормі, нуль: вимкнути
]

fis = MamdaniFIS('air_conditioner', [temperature, rate], [regulator], rules)
fis.print_structure()
print()
fis.print_rules()
fis.write_fis(out('air_conditioner.fis'))

plot_memberships([temperature, rate, regulator], out('task2_1_memberships.png'))

# Перевірка кожного правила
print("\nПеревірка правил (вхідні значення у вершинах термів):")
print(f"{'№':>2} {'T, °C':>6} {'dT/dt':>6} {'Кут, °':>8}  Терм висновку")
for i, r in enumerate(rules, 1):
    t = temperature.term(r.antecedent[0]).params[1]
    v = rate.term(r.antecedent[1]).params[1]
    angle = fis.evaluate([t, v])[0]
    print(f"{i:2d} {t:6.0f} {v:6.0f} {angle:8.2f}  {regulator.term(r.consequent[0]).label}")

#  Приклади виводу
print("\nПриклади нечіткого виводу:")
print(f"{'T, °C':>6} {'dT/dt':>6} {'Кут, °':>8}")
for t, v in [(32, 0.8), (30, -0.5), (26, 0.4), (25, -0.6), (22, 0.0), (23, 0.3), (20, -0.3), (17, 0.5), (13, -0.7)]:
    print(f"{t:6d} {v:6.1f} {fis.evaluate([t, v])[0]:8.2f}")

point = [26, 0.4]
plot_rule_viewer(fis, point, out('task2_2_rule_viewer.png'))

x1, x2, y = fis.surface(n=21)
plot_surface(x1, x2, y, ('Температура, °C', 'Швидкість, °C/хв', 'Кут, °'),
             'Кут повороту регулятора кондиціонера', out('task2_3_surface.png'))

plt.show()
