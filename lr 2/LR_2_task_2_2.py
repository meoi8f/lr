import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 3000

with open(input_file, 'r') as f:
    for line in f:
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
print("Дані завантажено:", X.shape)

label_encoders = []
X_encoded = np.empty(X.shape)

for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoders.append((i, encoder))

X_data = X_encoded[:, :-1].astype(int)
y_data = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=5
)

print("Дані розділено")
print("Починається навчання моделі...")

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

print("Навчання завершено")

y_pred = model.predict(X_test)

print("Гаусове ядро RBF")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("Recall   :", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1-score :", round(f1_score(y_test, y_pred, average='weighted'), 4))