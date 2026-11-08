import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# Завантаження та очищення даних
data = pd.read_csv('renfe_small.csv')
print("Розмір набору:", data.shape)
data = data.dropna(subset=['price', 'train_class', 'fare'])
print("Після видалення пропусків:", data.shape)
print(data['price'].describe().round(2))

plt.figure()
plt.hist(data['price'], bins=40, edgecolor='black')
plt.xlabel('Ціна, євро')
plt.ylabel('Кількість квитків')
plt.title('Розподіл цін на квитки')
plt.savefig('task4_price_hist.png', dpi=150)


# Апостеріорний розподіл середньої ціни (нормальне апріорне)
def posterior(prices, prior_mean=60.0, prior_std=30.0):
    n = len(prices)
    variance = 1 / (1 / prior_std ** 2 + n / prices.var())
    mean = variance * (prior_mean / prior_std ** 2 + n * prices.mean() / prices.var())
    return mean, np.sqrt(variance)


mean, std = posterior(data['price'])
low, high = stats.norm.interval(0.95, mean, std)
print(f"\nАпостеріорна середня ціна: {mean:.2f} євро, 95% інтервал [{low:.2f}; {high:.2f}]")

print("\nАпостеріорна середня ціна за типом потяга:")
results = {}
for train in ['AVE', 'ALVIA', 'INTERCITY', 'AV City']:
    results[train] = posterior(data.loc[data['train_type'] == train, 'price'])
    print(f"{train:<10} {results[train][0]:7.2f}  std = {results[train][1]:.3f}")


def prob_greater(a, b):
    return stats.norm.cdf((a[0] - b[0]) / np.sqrt(a[1] ** 2 + b[1] ** 2))


print("\nP(середня ціна AVE > ALVIA) =", round(prob_greater(results['AVE'], results['ALVIA']), 4))
print("P(середня ціна ALVIA > AV City) =", round(prob_greater(results['ALVIA'], results['AV City']), 4))

plt.figure()
for train, (m, s) in results.items():
    x = np.linspace(m - 4 * s, m + 4 * s, 200)
    plt.plot(x, stats.norm.pdf(x, m, s), label=train)
plt.xlabel('Середня ціна, євро')
plt.ylabel('Щільність')
plt.title('Апостеріорні розподіли середньої ціни')
plt.legend()
plt.savefig('task4_posteriors.png', dpi=150)

# Теорема Байєса: P(дорогий | тариф)
median_price = data['price'].median()
data['expensive'] = (data['price'] > median_price).astype(int)
print("\nМедіана ціни:", median_price)

p_expensive = data['expensive'].mean()
print(f"P(дорогий) = {p_expensive:.3f}")
for fare in ['Promo', 'Flexible', 'Adulto ida', 'Promo +']:
    p_fare = (data['fare'] == fare).mean()
    p_fare_given = (data.loc[data['expensive'] == 1, 'fare'] == fare).mean()
    print(f"P(дорогий | {fare}) = {p_fare_given * p_expensive / p_fare:.3f}")

# Наївний байєсівський класифікатор
features = ['origin', 'destination', 'train_type', 'train_class', 'fare']
encoder = OrdinalEncoder()
X = encoder.fit_transform(data[features]).astype(int)
y = data['expensive'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

classifier = CategoricalNB(min_categories=[len(c) for c in encoder.categories_])
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("\nCategoricalNB, прогноз дорогого квитка")
print("Accuracy =", round(accuracy_score(y_test, y_pred), 3))
print("Частка більшого класу =", round(max(y_test.mean(), 1 - y_test.mean()), 3))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Дешевий', 'Дорогий']))

# Прогноз для нового квитка
tickets = pd.DataFrame([['MADRID', 'BARCELONA', 'AVE', 'Turista', 'Promo'],
                        ['MADRID', 'SEVILLA', 'AVE', 'Preferente', 'Flexible'],
                        ['MADRID', 'VALENCIA', 'REGIONAL', 'Turista', 'Adulto ida']], columns=features)
probabilities = classifier.predict_proba(encoder.transform(tickets).astype(int))
for ticket, probability in zip(tickets.values, probabilities):
    print(ticket, '-> P(дорогий) =', round(probability[1], 3))

plt.show()
