import os
import time

import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, 'income_data.txt')

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        line = line.strip()
        if not line:
            continue

        data = line.split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
print(f"Розмір набору: {X.shape}, класи: <=50K - {count_class1}, >50K - {count_class2}")

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', make_pipeline(preprocessing.StandardScaler(), LogisticRegression(solver='liblinear'))))
models.append(('LDA', make_pipeline(preprocessing.StandardScaler(), LinearDiscriminantAnalysis())))
models.append(('KNN', make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier())))
models.append(('CART', make_pipeline(preprocessing.StandardScaler(), DecisionTreeClassifier(random_state=1))))
models.append(('NB', make_pipeline(preprocessing.StandardScaler(), GaussianNB())))
models.append(('SVM', make_pipeline(preprocessing.StandardScaler(), SVC(gamma='auto'))))

# 10-кратна стратифікована крос-валідація
results = []
names = []
print("\nТочність (accuracy), 10-кратна крос-валідація: середнє (std)")
for name, model in models:
    start = time.time()
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)   [%.1f c]' % (name, cv_results.mean(), cv_results.std(), time.time() - start))

pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison (income_data)')
pyplot.ylabel('accuracy')
pyplot.tight_layout()
pyplot.savefig(os.path.join(BASE_DIR, "task4_algorithm_comparison.png"), dpi=150)
pyplot.show()

# Показники якості на контрольній вибірці
print("\nПоказники якості на контрольній вибірці (20%), усереднення weighted:")
print("%-5s %9s %10s %8s %8s" % ("Model", "Accuracy", "Precision", "Recall", "F1"))
scores = {}
for name, model in models:
    model.fit(X_train, Y_train)
    pred = model.predict(X_validation)
    acc = accuracy_score(Y_validation, pred)
    prec = precision_score(Y_validation, pred, average='weighted')
    rec = recall_score(Y_validation, pred, average='weighted')
    f1 = f1_score(Y_validation, pred, average='weighted')
    scores[name] = (acc, prec, rec, f1)
    print("%-5s %9.4f %10.4f %8.4f %8.4f" % (name, acc, prec, rec, f1))

# Вибір найкращої моделі за F1-мірою
best_name = max(scores, key=lambda n: scores[n][3])
best_model = dict(models)[best_name]
print(f"\nНайкраща модель за F1: {best_name}")

pred = best_model.predict(X_validation)
target_names = label_encoder[-1].classes_
print(confusion_matrix(Y_validation, pred))
print(classification_report(Y_validation, pred, target_names=target_names))

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
              'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([item])[0])
        count += 1

predicted_class = best_model.predict(np.array(input_data_encoded).reshape(1, -1))
print(f"Тестова точка ({best_name}) належить до класу: "
      + label_encoder[-1].inverse_transform(predicted_class)[0])
