import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


#  Функції належності
def trimf(x, a, b, c):
    """Трикутна функція належності з параметрами [a b c]."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if a < b:
        rising = (a < x) & (x < b)
        y[rising] = (x[rising] - a) / (b - a)
    if b < c:
        falling = (b < x) & (x < c)
        y[falling] = (c - x[falling]) / (c - b)
    y[x == b] = 1.0
    return y


def gaussmf(x, sigma, c):
    """Гаусова функція належності з параметрами [sigma c]."""
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))


MF_TYPES = {'trimf': trimf, 'gaussmf': gaussmf}


class Term:
    """Терм лінгвістичної змінної."""

    def __init__(self, name, mf_type, params, label=None):
        self.name = name
        self.mf_type = mf_type
        self.params = list(params)
        self.label = label or name

    def mu(self, x):
        return MF_TYPES[self.mf_type](x, *self.params)


class Variable:
    """Лінгвістична змінна: назва, діапазон і терм-множина."""

    def __init__(self, name, value_range, terms):
        self.name = name
        self.range = tuple(value_range)
        self.terms = list(terms)

    def term(self, name):
        for t in self.terms:
            if t.name == name:
                return t
        raise KeyError(f"У змінної {self.name} немає терму {name}")

    def index(self, name):
        return [t.name for t in self.terms].index(name) + 1


def uniform_trimf(value_range, names, labels=None):
    """N трикутних термів рівномірно по діапазону (як команда Add MFs у MATLAB)."""
    lo, hi = value_range
    n = len(names)
    step = (hi - lo) / (n - 1)
    labels = labels or names
    return [Term(names[i], 'trimf', [lo + (i - 1) * step, lo + i * step, lo + (i + 1) * step], labels[i])
            for i in range(n)]


def uniform_gaussmf(value_range, names, labels=None):
    """N гаусових термів рівномірно по діапазону (як команда Add MFs у MATLAB)."""
    lo, hi = value_range
    n = len(names)
    step = (hi - lo) / (n - 1)
    sigma = step / 2.0 / np.sqrt(2.0 * np.log(2.0))
    labels = labels or names
    return [Term(names[i], 'gaussmf', [sigma, lo + i * step], labels[i]) for i in range(n)]


# Система нечіткого виводу
class Rule:
    """Правило: терми входів, терми виходів, вага."""

    def __init__(self, antecedent, consequent, weight=1.0):
        self.antecedent = list(antecedent)
        self.consequent = list(consequent)
        self.weight = weight


class MamdaniFIS:
    def __init__(self, name, inputs, outputs, rules, n_points=101):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.rules = rules
        self.n_points = n_points
        for r in rules:  # перевірка, що всі терми правил існують
            for v, t in zip(inputs, r.antecedent):
                v.term(t)
            for v, t in zip(outputs, r.consequent):
                v.term(t)

    def evaluate(self, values, details=False):
        """Обчислює чіткі значення виходів для вектора входів."""
        values = [float(np.clip(x, *v.range)) for x, v in zip(values, self.inputs)]
        # 1. Фазифікація та рівень істинності передумови кожного правила (AND = min)
        firing = []
        for r in self.rules:
            degrees = [float(v.term(t).mu(np.array([x]))[0])
                       for v, t, x in zip(self.inputs, r.antecedent, values)]
            firing.append(min(degrees) * r.weight)

        results, grids, aggregated = [], [], []
        for k, out in enumerate(self.outputs):
            grid = np.linspace(out.range[0], out.range[1], self.n_points)
            total = np.zeros_like(grid)
            # 2. Імплікація (min) і 3. агрегація висновків (max)
            for r, w in zip(self.rules, firing):
                total = np.maximum(total, np.minimum(w, out.term(r.consequent[k]).mu(grid)))
            # 4. Дефазифікація методом центра ваги
            area = total.sum()
            if area > 0:
                results.append(float((grid * total).sum() / area))
            else:
                # Жодне правило не спрацювало: MATLAB повертає середину діапазону
                results.append((out.range[0] + out.range[1]) / 2.0)
            grids.append(grid)
            aggregated.append(total)
        if details:
            return results, firing, grids, aggregated
        return results

    def surface(self, out_index=0, n=15):
        """Поверхня «входи-вихід» на сітці n x n (для системи з двома входами)."""
        x1 = np.linspace(*self.inputs[0].range, n)
        x2 = np.linspace(*self.inputs[1].range, n)
        y = np.zeros((n, n))
        for i, b in enumerate(x2):
            for j, a in enumerate(x1):
                y[i, j] = self.evaluate([a, b])[out_index]
        return x1, x2, y

    # Текстове подання
    def print_rules(self):
        print(f"База правил системи '{self.name}' ({len(self.rules)}):")
        for i, r in enumerate(self.rules, 1):
            left = ' and '.join(f"({v.name} is {v.term(t).label})" for v, t in zip(self.inputs, r.antecedent))
            right = ', '.join(f"({v.name} is {v.term(t).label})" for v, t in zip(self.outputs, r.consequent))
            print(f"{i:2d}. If {left} then {right} ({r.weight:g})")

    def print_structure(self):
        print(f"Система '{self.name}': Мамдані, AND=min, імплікація=min, агрегація=max, дефазифікація=centroid")
        for kind, group in (("Вхід", self.inputs), ("Вихід", self.outputs)):
            for v in group:
                print(f"{kind} {v.name}, діапазон [{v.range[0]:g}, {v.range[1]:g}]:")
                for t in v.terms:
                    params = ' '.join(f"{p:.4g}" for p in t.params)
                    print(f"   {t.name:<4} {t.label:<22} {t.mf_type}  [{params}]")

    def write_fis(self, path):
        """Зберігає систему у форматі .fis, який відкривається в MATLAB (команда fuzzy)."""
        def fmt(values):
            return ' '.join(f"{p:.6g}" for p in values)

        lines = ["[System]", f"Name='{self.name}'", "Type='mamdani'", "Version=2.0",
                 f"NumInputs={len(self.inputs)}", f"NumOutputs={len(self.outputs)}", f"NumRules={len(self.rules)}",
                 "AndMethod='min'", "OrMethod='max'", "ImpMethod='min'", "AggMethod='max'",
                 "DefuzzMethod='centroid'", ""]
        for section, group in (("Input", self.inputs), ("Output", self.outputs)):
            for i, v in enumerate(group, 1):
                lines += [f"[{section}{i}]", f"Name='{v.name}'", f"Range=[{fmt(v.range)}]", f"NumMFs={len(v.terms)}"]
                lines += [f"MF{j}='{t.name}':'{t.mf_type}',[{fmt(t.params)}]" for j, t in enumerate(v.terms, 1)]
                lines.append("")
        lines.append("[Rules]")
        for r in self.rules:
            left = ' '.join(str(v.index(t)) for v, t in zip(self.inputs, r.antecedent))
            right = ' '.join(str(v.index(t)) for v, t in zip(self.outputs, r.consequent))
            lines.append(f"{left}, {right} ({r.weight:g}) : 1")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


# Графіки
def plot_memberships(variables, path, title=None):
    """Функції належності змінних (аналог Membership Function Editor)."""
    fig, axes = plt.subplots(len(variables), 1, figsize=(9, 2.9 * len(variables)), squeeze=False)
    for ax, v in zip(axes[:, 0], variables):
        x = np.linspace(v.range[0], v.range[1], 501)
        for t in v.terms:
            y = t.mu(x)
            ax.plot(x, y, lw=1.8, label=t.label)
        ax.set_xlim(*v.range)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel(f"змінна {v.name}")
        ax.set_ylabel("ступінь належності")
        ax.grid(alpha=.3)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8, frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_rule_viewer(fis, values, path):
    """Візуалізація нечіткого виводу (аналог Rule Viewer)."""
    results, firing, grids, aggregated = fis.evaluate(values, details=True)
    variables = fis.inputs + fis.outputs
    n_in, n_rules = len(fis.inputs), len(fis.rules)
    rows = n_rules + 1
    fig, axes = plt.subplots(rows, len(variables), figsize=(2.9 * len(variables), 0.62 * rows + 1.0),
                             squeeze=False, sharex='col')
    for c, v in enumerate(variables):
        x = np.linspace(v.range[0], v.range[1], 301)
        for r, rule in enumerate(fis.rules):
            ax = axes[r, c]
            name = (rule.antecedent + rule.consequent)[c]
            y = v.term(name).mu(x)
            ax.plot(x, y, color='#555555', lw=.8)
            if c < n_in:
                level = min(1.0, float(v.term(name).mu(np.array([values[c]]))[0]))
                ax.fill_between(x, 0, np.minimum(y, level), color='#f2d21b')
                ax.axvline(values[c], color='red', lw=1)
            else:
                ax.fill_between(x, 0, np.minimum(y, firing[r]), color='#3b5bdb')
            ax.set_ylim(0, 1.05)
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(str(r + 1), rotation=0, labelpad=10, va='center')
        ax = axes[n_rules, c]
        ax.set_yticks([])
        ax.set_ylim(0, 1.05)
        if c < n_in:
            ax.axis('off')
            axes[0, c].set_title(f"{v.name} = {values[c]:g}", fontsize=10)
        else:
            k = c - n_in
            ax.fill_between(grids[k], 0, aggregated[k], color='#3b5bdb')
            ax.axvline(results[k], color='red', lw=2)
            axes[0, c].set_title(f"{v.name} = {results[k]:.3g}", fontsize=10)
            ax.tick_params(labelbottom=True, labelsize=8)
    fig.tight_layout()
    fig.subplots_adjust(hspace=.12)
    fig.savefig(path, dpi=150)
    return fig


def plot_surface(x1, x2, y, labels, title, path):
    """Тривимірна поверхня."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    xx, yy = np.meshgrid(x1, x2)
    ax.plot_surface(xx, yy, y, cmap=cm.viridis, edgecolor='k', linewidth=.3)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig
