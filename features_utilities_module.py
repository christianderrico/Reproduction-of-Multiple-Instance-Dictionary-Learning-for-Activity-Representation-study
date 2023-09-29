import os
from abc import abstractmethod, ABC

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

'''
This is a singleton object useful for managing videos and the features/information extracted from videos.

Attributes:
- `videos_path` (str): Path to the folder containing videos.
- `vidnum` (int): Number of videos.
- `cname` (list of str): Categories names.
- `label` (list of int): Class labels for each video.
- `features` (list): Features extracted by Laptev STIP extractor.
- `xyt` (list): Spatial and temporal coordinates associated with features.
- `path` (str): Path to the specific analyzed video.
- `nclass` (int): Number of different classes.
'''


class DbHelper(object):

    def _create_path_name(self, v):
        return os.path.join(self._videos_path, v)

    """
    This extracts a video specific prop calling VideoCapture from OpenCV
    """
    def _extract_infos(self, videos, prop_key):
        return [cv2.VideoCapture(v).get(prop_key) for v in videos]

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

        videos = self.get_all_videos()
        self.nframe_videos, self.width_videos, self.height_videos = \
            [self._extract_infos(videos, k)
             for k in [cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]]

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
        self.tr = self.get_training_set()

    def create_tr(self, idx):
        tr = np.zeros(len(self.path))
        tr[idx] = 1
        return tr

    @abstractmethod
    def get_all_videos(self):
        pass

    @abstractmethod
    def get_training_set(self):
        pass


'''
This extends DbHelper in order to manage KTH Dataset and its data structure.
'''


class KTHHelper(DbHelper, ABC):

    def get_all_videos(self):
        return [self._create_path_name(v) for v in os.listdir(self._videos_path)
                if not os.path.isdir(self._create_path_name(v))]


    def get_training_set(self):

        """
        For the test phase, subjects with IDs 2, 3, 5, 6, 7, 8, 9, 10, and 22 are selected,
        as indicated in the paper, while the others are involved in the model training procedure
        :return: training set
        """
        test_set = [2, 3, 5, 6, 7, 8, 9, 10, 22]
        names = ["person" + ("0{}".format(v) if v < 10 else str(v)) for v in test_set]
        id_tr = [i for i, path in enumerate(self.path) if not any(name in os.path.basename(path) for name in names)]

        return self.create_tr(id_tr)


"""
    For completeness, code for the Hollywood2 dataset has been prepared, 
    but the STIP extractor hasn't been able to gather sufficient information from those videos; 
    therefore, it hasn't been possible to work with this dataset 
    and apply the same approach used with the KTH dataset.
"""


class HollywoodHelper(DbHelper, ABC):

    def get_all_videos(self):
        videos_path = [os.path.join(self._videos_path, d) for d in os.listdir(self._videos_path)]
        files = [os.listdir(v) for v in videos_path]
        return [os.path.join(videos_path[i], f) for i in range(len(videos_path)) for f in files[i]]

    def get_training_set(self):
        tr_idx, _ = train_test_split(range(len(self.label)), test_size=0.3, stratify=self.label, random_state=42)
        return self.create_tr(tr_idx)


def load_features(file):
    """
    Load features from files produced by the STIP extractor at the end of the extraction procedure.

    These features are stored in a .txt file that contains information about the frame number
    and spatial location of the keypoint, along with the concatenated HOG/HOF descriptors.

    :param file: The file of interest containing all the information (str).

    Returns:
        xyt: spatial-temporal coordinates
        features: HOG/HOF features
    """
    import chardet
    # position = np.genfromtxt(file, comments='#', usecols=(4, 5, 6, 7, 8), dtype=np.int32)
    descriptors = np.genfromtxt(file, comments='#')

    if descriptors.size > 0:
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
    else:
        print("File: ", file)
        xyt = []
        features = []

    return xyt, features
