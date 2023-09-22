import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def data_split(x, y, db):
    train_indexes = np.where(db.tr == 1)[0]
    test_indexs = [i for i in range(len(x)) if i not in train_indexes]
    x_train, y_train = x[train_indexes], y[train_indexes]
    x_test, y_test = x[test_indexs], y[test_indexs]

    return x_train, y_train, x_test, y_test


def plot_confusion_matrix(matrix, class_names):

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted labels")
    plt.ylabel("Real labels")
    plt.show()


def test_approach(x, db):
    y = db.label
    x_train, y_train, x_test, y_test = data_split(x, y, db)

    model = svm.SVC(C=20)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy is: ", accuracy)

    # Calcola la matrice di confusione
    cf_matrix = confusion_matrix(y_test, y_pred)
    class_names = [db.id_class2name[id_label] for id_label in np.unique(y)]

    plot_confusion_matrix(cf_matrix, class_names)
