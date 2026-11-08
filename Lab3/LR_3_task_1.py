import itertools
import os

import numpy as np
import matplotlib.pyplot as plt

from fuzzy_mamdani import MamdaniFIS, Rule, Variable, plot_memberships, plot_rule_viewer, plot_surface, uniform_trimf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(BASE_DIR, name)


#  Лінгвістичні змінні
# Входи. Діапазони в умові не задані, тому температуру взято у °C, а напір - у % від максимального.
temperature = Variable('temperature', (0, 100), uniform_trimf(
    (0, 100), ['C', 'CL', 'W', 'NH', 'H'],
    ['Холодна', 'Прохолодна', 'Тепла', 'Не дуже гаряча', 'Гаряча']))
pressure = Variable('pressure', (0, 100), uniform_trimf(
    (0, 100), ['WK', 'NS', 'S'], ['Слабий', 'Не дуже сильний', 'Сильний']))

# Виходи: кут повороту крана, градуси. Вправо (+) - збільшити потік, вліво (-) - зменшити.
ANGLE_TERMS = ['LB', 'LM', 'LS', 'Z', 'RS', 'RM', 'RB']
ANGLE_LABELS = ['Великий вліво', 'Середній вліво', 'Невеликий вліво', 'Без змін',
                'Невеликий вправо', 'Середній вправо', 'Великий вправо']
hot_tap = Variable('hot_tap', (-90, 90), uniform_trimf((-90, 90), ANGLE_TERMS, ANGLE_LABELS))
cold_tap = Variable('cold_tap', (-90, 90), uniform_trimf((-90, 90), ANGLE_TERMS, ANGLE_LABELS))

# База правил
rules = [
    Rule(['H', 'S'], ['LM', 'RM']),   # 1. Гаряча, сильний: гарячий - середній вліво, холодний - середній вправо
    Rule(['H', 'NS'], ['Z', 'RM']),   # 2. Гаряча, не дуже сильний: холодний - середній вправо
    Rule(['NH', 'S'], ['LS', 'Z']),   # 3. Не дуже гаряча, сильний: гарячий - невеликий вліво
    Rule(['NH', 'WK'], ['RS', 'RS']), # 4. Не дуже гаряча, слабий: обидва - невеликий вправо
    Rule(['W', 'NS'], ['Z', 'Z']),    # 5. Тепла, не дуже сильний: без змін
    Rule(['CL', 'S'], ['RM', 'LM']),  # 6. Прохолодна, сильний: гарячий - середній вправо, холодний - середній вліво
    Rule(['CL', 'NS'], ['RM', 'LS']), # 7. Прохолодна, не дуже сильний: гарячий - середній вправо, холодний - невеликий вліво
    Rule(['C', 'WK'], ['RB', 'Z']),   # 8. Холодна, слабий: гарячий - великий вправо
    Rule(['C', 'S'], ['LM', 'RM']),   # 9. Холодна, сильний: гарячий - середній вліво, холодний - середній вправо
    Rule(['W', 'S'], ['LS', 'LS']),   # 10. Тепла, сильний: обидва - невеликий вліво
    Rule(['W', 'WK'], ['RS', 'RS']),  # 11. Тепла, слабий: обидва - невеликий вправо
]

fis = MamdaniFIS('water_taps', [temperature, pressure], [hot_tap, cold_tap], rules)
fis.print_structure()
print()
fis.print_rules()
fis.write_fis(out('water_taps.fis'))

plot_memberships([temperature, pressure], out('task1_1_inputs.png'))
plot_memberships([hot_tap, cold_tap], out('task1_2_outputs.png'))
print("\nПеревірка правил (вхідні значення у вершинах термів):")
print(f"{'№':>2} {'T, °C':>6} {'Напір,%':>8} {'Гарячий':>9}  {'Холодний':>9}  Терми висновку")
for i, r in enumerate(rules, 1):
    t = temperature.term(r.antecedent[0]).params[1]
    p = pressure.term(r.antecedent[1]).params[1]
    hot, cold = fis.evaluate([t, p])
    print(f"{i:2d} {t:6.0f} {p:8.0f} {hot:9.2f}  {cold:9.2f}  {hot_tap.term(r.consequent[0]).label}; "
          f"{cold_tap.term(r.consequent[1]).label}")

covered = {tuple(r.antecedent) for r in rules}
missing = [(a.label, b.label) for a, b in itertools.product(temperature.terms, pressure.terms)
           if (a.name, b.name) not in covered]
print(f"\nПоєднання термів, для яких в умові немає правил ({len(missing)}):")
for a, b in missing:
    print(f"   температура «{a}», напір «{b}»")

# Приклади виводу
print("\nПриклади нечіткого виводу:")
print(f"{'T, °C':>6} {'Напір,%':>8} {'Гарячий':>9} {'Холодний':>9}")
for t, p in [(90, 90), (80, 60), (60, 75), (50, 50), (35, 80), (30, 45), (10, 15), (65, 20)]:
    hot, cold = fis.evaluate([t, p])
    print(f"{t:6d} {p:8d} {hot:9.2f} {cold:9.2f}")

point = [70, 80]
plot_rule_viewer(fis, point, out('task1_3_rule_viewer.png'))

#  Поверхні «входи-вихід»
for k, (var, name) in enumerate([(hot_tap, 'гарячої'), (cold_tap, 'холодної')]):
    x1, x2, y = fis.surface(out_index=k, n=21)
    plot_surface(x1, x2, y, ('Температура, °C', 'Напір, %', 'Кут, °'),
                 f"Кут повороту крана {name} води", out(f"task1_{4 + k}_surface_{var.name}.png"))

plt.show()
