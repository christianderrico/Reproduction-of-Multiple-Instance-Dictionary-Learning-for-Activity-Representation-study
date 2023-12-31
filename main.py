# This is a sample Python script.
import os
import random

import joblib
import numpy as np
from features_utilities_module import DbHelper, load_features, KTHHelper, HollywoodHelper
from encoding_module import extract_codebooks, videos_encoding
from model_training_module import extract_codebook
from testing_module import test_approach


def rand_permutation(vect):
    vect = np.array(vect.copy())
    ri = np.random.permutation(len(vect)).tolist()

    vect = vect[ri]

    return vect


def get_bag_features_and_labels(db, action_class):
    print("Loading features for class {0}: ".format(action_class))
    pos_bags = [id for id in np.where(db.label == action_class)[0] if db.tr[id] == 1]
    neg_bags = [id for id in np.where(db.label != action_class)[0] if db.tr[id] == 1]

    pos_bags = rand_permutation(pos_bags)
    neg_bags = rand_permutation(neg_bags)

    bags = np.concatenate((pos_bags, neg_bags)).astype(int)

    bag_features = np.empty(len(bags), dtype=object)
    labels = np.empty(len(bags), dtype=object)

    for j, video_id in enumerate(bags):
        feat = db.features[video_id]
        # xy, feat = load_features(db['path'][video_id])

        if db.label[video_id] == action_class:
            cur_l = 1
            #limit = 500
        else:
            cur_l = -1
            #limit = 300

        limit = 10_000

        len_feat = feat.shape[0]

        if len_feat > limit:
            idx = np.random.choice(len_feat, limit)
            feat = feat[idx, :]
        else:
            ri = np.random.permutation(len_feat)
            feat = feat[ri, :]

        bag_features[j] = feat
        labels[j] = np.ones(feat.shape[0]) * cur_l

    feature_labels = np.array([np.unique(l)[0] for l in labels])

    return bag_features, feature_labels


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    videos_path = r".\all_videos"
    features_path = os.path.join(videos_path, "descr")
    # path = r"D:\pythonProject5\Hollywood"
    # videos_path = path + "\\all_videos"
    # features_path = path + "\\all_descr"

    db = KTHHelper(videos_path, features_path)
    #db = HollywoodHelper(videos_path, features_path)

    codebooks = np.zeros(db.nclass, dtype=object)
    for reference_class in range(db.nclass):
        codebook_path = r".\codebooks\codebook_{0}.pkl".format(reference_class)
        if os.path.exists(codebook_path):
            codebook = joblib.load(codebook_path)
        else:
            bags, feature_labels = get_bag_features_and_labels(db, reference_class)
            codebook = extract_codebook(bags, feature_labels, reference_class)

        codebooks[reference_class] = codebook

    fv = videos_encoding(db, codebooks)
    test_approach(fv, db)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
