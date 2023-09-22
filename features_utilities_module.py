import os
import cv2
import numpy as np


class DbHelper(object):

    def _create_path_name(self, v):
        return os.path.join(self._videos_path, v)

    def __init__(self, videos_path, directory_data):
        subfolders = os.listdir(directory_data)
        self._videos_path = videos_path

        self.vidnum = 0
        self.cname = []
        self.label = []
        self.features = []
        self.xyt = []
        self.path = []
        self.nclass = 0

        videos = [self._create_path_name(v) for v in os.listdir(videos_path) if not os.path.isdir(self._create_path_name(v))]
        self.nframe_videos = [int(cv2.VideoCapture(v).get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos]
        self.id_class2name = {int(i): name for i, name in enumerate(subfolders)}

        for i in range(len(subfolders)):
            subname = subfolders[i]
            self.cname.append(subname)

            videos = os.listdir(os.path.join(directory_data, subname))
            c_num = len(videos)

            self.vidnum += c_num
            self.label.append(np.ones((c_num, 1)) * self.nclass)

            self.nclass += 1

            for v in range(c_num):
                c_path = os.path.join(directory_data, subname, videos[v])
                xyt, features = load_features(c_path)
                self.xyt.append(xyt)
                self.features.append(features)
                self.path.append(c_path)

        self.label = np.ravel(np.concatenate(self.label))
        self._get_training_set()

    def _get_training_set(self):
        training_set = [2, 3, 5, 6, 7, 8, 9, 10, 22]
        names = ["person" + ("0{}".format(v) if v < 10 else str(v)) for v in training_set]
        tr = [i for i, path in enumerate(self.path) for name in names if name in path]
        self.tr = np.zeros(len(self.path))
        self.tr[tr] = 1


def load_features(file):
    #position = np.genfromtxt(file, comments='#', usecols=(4, 5, 6, 7, 8), dtype=np.int32)
    descriptors = np.genfromtxt(file, comments='#')

    x, *y = descriptors.shape

    if not y:
        descriptors = np.reshape(descriptors, (1, x))

    # Il formato del dato originale è questo:
    # point-type x y t sigma2 tau2 detector-confidence dscr-hog(72) dscr-hof(90)

    xyt = descriptors[:, [1, 2, 3]]
    features = descriptors[:, 7:]

    xyt = list(enumerate(xyt))

    index = 1
    t = 2
    x_coord = 0
    y_coord = 1
    ids, elems = zip(*sorted(xyt, key=lambda x: (x[index][t], x[index][x_coord], x[index][y_coord])))
    xyt = list(elems)
    features = features[list(ids)]

    return xyt, features