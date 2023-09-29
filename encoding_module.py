import multiprocessing
import concurrent.futures

import numpy as np
import numpy.linalg


def videos_encoding(db, codebooks):
    """
    This realizes features encoding + pooling in order to represent videos using Bag-of-Words (BoWs).
    LLC is used to represent each feature vector as a combination of multiple elements in the codebook, and
    a spatio-temporal pyramid is used to pool multiple codes from each subregion.

    :param db: DbHelper object
    :param codebooks: extracted dictionaries
    :return: values associated to encoded videos
    """
    # concatenation of codebooks
    w = extract_codebooks(codebooks)
    # The video is first viewed as a whole, then, in the second level it is segmented into 4 subregions without
    # any temporal segmentation.
    # In the third level each part in the previous level is partitioned into 4 subregions in spatially
    # and 2 subregions in temporally
    pyramid = [1, 2, 2]
    temporal = [1, 1, 2]
    dSize = w.shape[0]

    def encoding_video_features(id_video):
        """
        Realizes the Spatio-temporal max pooling operation in order to maintain spatial
        information by pooling in local spatial and temporal bins the LLC encoded features

        :param id_video: ID of the video
        :return: Pooled representation of a video
        """
        print("Id video: ", id_video)
        xyt, features = db.xyt[id_video], db.features[id_video]
        # llc coding
        llc = llc_coding_appr(w, features)
        llc = llc.T

        pyramid_levels = len(pyramid)
        space_pooling_bins = np.square(pyramid)
        pyramid_bins = np.multiply(space_pooling_bins, temporal)
        total_bins = np.sum(pyramid_bins)

        betas = np.zeros((dSize, total_bins), dtype=object)
        global_bin_id = 0

        frame_width = db.width_videos[id_video]
        frame_height = db.height_videos[id_video]
        n_frames = db.nframe_videos[id_video]
        for n_level in range(pyramid_levels):
            nbins = pyramid_bins[n_level]
            wUnit = frame_width / pyramid[n_level]
            hUnit = frame_height / pyramid[n_level]
            temporalUnit = n_frames / temporal[n_level]

            xs, ys, ts = zip(*xyt)
            xBin = np.ceil(np.array(xs) / wUnit)
            yBin = np.ceil(np.array(ys) / hUnit)
            idxBin = (yBin - 1) * pyramid[n_level] + xBin

            if temporal[n_level] > 1:
                timeBin = np.clip(np.array(ts) // temporalUnit, 1, temporal[n_level])
                max_w, max_h = 2, 2
                idxBin = (timeBin - 1) * (max_w * max_h) + idxBin

            print("Ids level {0}: {1}".format(n_level, np.unique(idxBin)))

            for local_bin_id in range(nbins):
                sidxbin = np.where(idxBin == local_bin_id + 1)[0]
                if sidxbin.size != 0:
                    betas[:, global_bin_id] = np.max(llc[:, sidxbin], axis=1)
                global_bin_id = global_bin_id + 1

        assert global_bin_id == total_bins

        betas = betas.flatten()
        betas = betas / np.sqrt(np.sum(betas ** 2))

        return betas

    num_worker = int(multiprocessing.cpu_count() / 2 - 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        futures = [executor.submit(encoding_video_features, id_video) for id_video in range(db.vidnum)]
        fv = np.array([f.result() for f in futures])

    return fv


def extract_codebooks(codebooks):
    return np.concatenate([book.cluster_centers_ for book in codebooks])


def llc_coding_appr(B, X, knn=5, beta=1e-4):
    """
    Performs LLC encoding technique, useful to locally code a video
    in order to obtain a compact and discriminative representation.

    :param B: Codebook, a set of prototypes or reference point for the LLC encoding
    :param X: Features video which the encoding will be applied on.
    :param knn: Number of considered neighbor; for every feature will be selected knn neighbors
    :param beta: Regularization coefficient
    :return: A vector of approximated points Y = X * B, LLC representation of the original features.
    """
    nbase = B.shape[0]
    nframe = X.shape[0]
    D = np.sum(X ** 2, axis=1, keepdims=True) - 2 * np.dot(X, B.T) + np.sum(B ** 2, axis=1)
    IDX = np.zeros(shape=(nframe, knn))

    for i in range(nframe):
        d = D[i]
        idx = np.argsort(d)
        IDX[i] = idx[:knn]

    II = np.identity(knn)
    Coeff = np.zeros(shape=(nframe, nbase))
    for i in range(nframe):
        idx = IDX[i]
        idx = idx.astype(int)
        z = B[idx] - np.tile(X[i], (knn, 1))
        C = np.dot(z, z.T)
        C = C + II * beta * np.trace(C)
        w = numpy.linalg.solve(C, np.ones((knn, 1)))
        w = w / sum(w)
        Coeff[i, idx] = w.T

    return Coeff
