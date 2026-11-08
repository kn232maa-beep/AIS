import os

import numpy as np
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_and_show(name):
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(OUT_DIR, name), dpi=150)
    pyplot.show()


# КРОК 1. ЗАВАНТАЖЕННЯ ТА ВИВЧЕННЯ ДАНИХ
iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
print()
print(iris_dataset['DESCR'][:193] + "\n...")
print("\nПовний опис набору даних (DESCR):")
print(iris_dataset['DESCR'])
print("Назви відповідей: {}".format(iris_dataset['target_names']))
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))
print("Значення ознак для перших п'яти прикладів:\n{}".format(iris_dataset['data'][:5]))
print("Тип масиву target: {}".format(type(iris_dataset['target'])))
print("Форма масиву target: {}".format(iris_dataset['target'].shape))
print("Відповіді:\n{}".format(iris_dataset['target']))

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = DataFrame(iris_dataset['data'], columns=names[:4])
dataset['class'] = ['Iris-' + iris_dataset['target_names'][t] for t in iris_dataset['target']]

# shape
print("\nФорма датасету:", dataset.shape)

# Зріз даних head
print(dataset.head(20))

# Статистичні зведення методом describe
print(dataset.describe())

# Розподіл за атрибутом class
print(dataset.groupby('class').size())

#  КРОК 2. ВІЗУАЛІЗАЦІЯ ДАНИХ
# Діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
save_and_show("task3_boxplot.png")

# Гістограма розподілу атрибутів датасета
dataset.hist()
save_and_show("task3_hist.png")

# Матриця діаграм розсіювання (колір точки визначається класом)
colors = dataset['class'].map({'Iris-setosa': 'tab:red',
                               'Iris-versicolor': 'tab:green',
                               'Iris-virginica': 'tab:blue'})
scatter_matrix(dataset, c=colors, figsize=(9, 9), alpha=0.8)
save_and_show("task3_scatter_matrix.png")

# КРОК 3. НАВЧАЛЬНИЙ ТА ТЕСТОВИЙ НАБОРИ
# Розділення датасету на навчальну та контрольну вибірки
array = dataset.values
X = array[:, 0:4].astype(float)
y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# КРОК 4. КЛАСИФІКАЦІЯ (ПОБУДОВА МОДЕЛІ)
models = []

models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=1)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

print("\nТочність (accuracy) моделей, 10-кратна крос-валідація: середнє (std)")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Порівняння алгоритмів
pyplot.figure()
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison')
save_and_show("task3_algorithm_comparison.png")

# КРОК 6. ПРОГНОЗ НА КОНТРОЛЬНІЙ ВИБІРЦІ
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# КРОК 7. ОЦІНКА ЯКОСТІ МОДЕЛІ
print("\nОцінка моделі SVM на контрольній вибірці:")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#  КРОК 8. ЗАСТОСУВАННЯ МОДЕЛІ ДЛЯ НОВИХ ДАНИХ
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

prediction = model.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозована мітка: {}".format(prediction[0].replace('Iris-', '')))
