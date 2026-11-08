# Завдання 2.5. Класифікація даних лінійним класифікатором Ridge
# (у методичці файл помилково названо LR_2_task_4.py — він уже зайнятий завданням 2.4)
# ======================================================================
# Приклад класифікатора Ridge
# ======================================================================
import os
from io import BytesIO  # потрібен для збереження графіка у "файловий об'єкт"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  # у методичці імпорт пропущено

sns.set()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

iris = load_iris()
X, y = iris.data, iris.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

# tol=1e-2   - точність (критерій зупинки) ітераційного розв'язувача
# solver='sag' - Stochastic Average Gradient, стохастичний середній градієнтний спуск
# alpha=1.0 (за замовчуванням) - сила L2-регуляризації
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)  # у методичці помилково X_test

print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))
# Коефіцієнт Коена Каппа - узгодженість прогнозу з істинними мітками з поправкою
# на випадковий збіг (1 - повна згода, 0 - як випадкове вгадування)
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
# Коефіцієнт кореляції Метьюза - кореляція між прогнозом та істиною
# з урахуванням усіх елементів матриці помилок (від -1 до 1)
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4))
# у методичці аргументи переставлені: спочатку істинні мітки, потім прогноз
print('\t\tClassification Report:\n',
      metrics.classification_report(ytest, ypred, target_names=iris.target_names))

mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "Confusion.jpg"))

# Save SVG in a fake file object.
f = BytesIO()
plt.savefig(f, format="svg")
plt.show()
