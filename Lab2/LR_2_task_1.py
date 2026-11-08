# Завдання 2.1. Класифікація за допомогою машин опорних векторів (SVM)
import os

import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл, який містить дані
input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'income_data.txt')

# Читання даних
X = []
y = []
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

print(f"Завантажено точок: <=50K - {count_class1}, >50K - {count_class2}")

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
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

# Масштабування ознак (без нього LinearSVC не збігається:
# fnlwgt ~ 10^5, а більшість інших ознак ~ 10^0..10^1)
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# Створення SVМ-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))

# Навчання класифікатора
classifier.fit(X, y)

# Розбиття даних на навчальний та тестовий набори 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Обчислення F-міри для SVМ-класифікатора (3-кратна перехресна перевірка)
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score (cross-validation, cv=3): " + str(round(100 * f1.mean(), 2)) + "%")

# Інші показники якості на тестовому наборі
print("\nПоказники якості на тестовому наборі (20%):")
print("Accuracy (акуратність): " + str(round(100 * accuracy_score(y_test, y_test_pred), 2)) + "%")
print("Precision (точність):   " + str(round(100 * precision_score(y_test, y_test_pred, average='weighted'), 2)) + "%")
print("Recall (повнота):       " + str(round(100 * recall_score(y_test, y_test_pred, average='weighted'), 2)) + "%")
print("F1 score:               " + str(round(100 * f1_score(y_test, y_test_pred, average='weighted'), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
              'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
input_data_encoded = scaler.transform(input_data_encoded)

# Використання класифікатора для кодованої точки даних та виведення результату
predicted_class = classifier.predict(input_data_encoded)
print("\nТестова точка належить до класу: " + label_encoder[-1].inverse_transform(predicted_class)[0])
