import os

import joblib
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC


def extract_codebook(bags, bags_labels, n_class):

    features = np.concatenate(bags)
    feature_labels = np.concatenate([bags_labels[i] * np.ones(len(v)) for i, v in enumerate(bags)])
    groups = [len(b) for b in bags]

    i = 0
    max_iter = 5
    converge = False

    c = np.ones(len(feature_labels)) / len(feature_labels)
    y = feature_labels

    while not converge:

        svmm = optimize_s(features, y, c)
        c, new_y = optimize_l(features, svmm, groups, bags_labels)

        y_diff = np.sum(np.abs(y - new_y))
        print("Class changes: {0}".format(y_diff))

        if i >= max_iter or y_diff == 0:
            converge = True

        y = new_y
        i += 1

    id_pos = np.where(y > 0)[0]
    print("Pos len: ", len(id_pos))
    x_pos = features[id_pos, :]
    y_pred = svmm.predict(x_pos)

    positive_prediction_ids = np.where(y_pred > 0)[0]
    x_pos = x_pos[positive_prediction_ids, :]

    k = 4000
    codebook = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(x_pos)
    save_model(codebook, n_class)

    return codebook


def optimize_s(x, y, c, sample_ratio=0.7):

    pos_ids = np.where(y > 0)[0]
    neg_ids = np.where(y < 0)[0]

    c_pos = c[pos_ids]
    c_pos = c_pos / np.sum(c_pos)
    x_pos = x[pos_ids, :]
    y_pos = y[pos_ids]
    x_neg = x[neg_ids, :]
    y_neg = y[neg_ids]

    sample_size = round(sample_ratio * len(c_pos))
    sample_indexes = np.random.choice(len(c_pos), size=sample_size, replace=True, p=c_pos)
    x_pos = x_pos[sample_indexes, :]
    y_pos = y_pos[sample_indexes]

    x_train = np.concatenate((x_pos, x_neg))
    y_train = np.concatenate((y_pos, y_neg))
    print("Fitting SVC model")
    print("x: ", x_train.shape)
    print("y: ", y_train.shape)

    print("Training started at: ", datetime.now())
    model = LinearSVC(C=1, class_weight='balanced', verbose=False).fit(x_train, y_train)
    print("Training finished at: ", datetime.now())

    return model


def features2bags(values, groups):
    result = []
    start = 0
    for g in groups:
        end = start + g
        result.append(values[start:end])
        start = end
    return result


def fix_classes(y, c, bags_size, bag_labels):
    result = []
    y2_bags = features2bags(y.copy(), bags_size)
    c_bags = features2bags(c.copy(), bags_size)

    for i, predicted_bag in enumerate(y2_bags):
        if bag_labels[i] == 1:
            arg_id = np.argmax(c_bags[i])
            predicted_bag[arg_id] = 1
        else:
            predicted_bag[:] = -1

        result.append(predicted_bag)

    return np.concatenate(result)


def optimize_l(x, svmm, groups, bag_labels, sigma=1):

    y2 = svmm.predict(x)
    print("Percentage of negative patches: ", len(y2[y2 < 0]) / len(y2))
    c = svmm.decision_function(x)

    result = fix_classes(y2, c, groups, bag_labels)
    c = 1 / (1 + np.exp(-c / sigma))

    return c, result


def save_model(model, n_class):
    folder = "codebooks/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = "codebook_{0}.pkl".format(n_class)
    print(folder + file_path)
    joblib.dump(model, folder + file_path)