from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from sklearn import feature_extraction
import os
from base.IR_feature_weighting import okapi_BM_25
# Put root project dir in a global constant
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Helper:
    def __init__(self):
        # Put root project dir in a constant
        # Loading of data using pandas, that creates a new object URM_data with the first line values as attributes
        # (playlist_id, track_id) as formatted in the .csv file
        self.URM_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/dataset.csv"))
        self.users_list_data = np.asarray(list(self.URM_data.row))
        self.items_list_data = np.asarray(list(self.URM_data.col))


    def convert_URM_to_csr(self, URM):
        ratings_list = np.ones(len(np.asarray(list(URM.data))))
        URM = sps.coo_matrix((ratings_list, (np.asarray(list(URM.row)), np.asarray(list(URM.col)))))
        URM = URM.tocsr()
        return URM

    def load_URM_train_csr(self):
        URM_train = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/train_data.csv"))
        URM_train = self.convert_URM_to_csr(URM_train)
        return URM_train

    def load_URM_test(self):
        URM_test = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/test_data.csv"))
        return URM_test

    def load_URM_test_csr(self):
        URM_test = self.load_URM_test()
        ratings_list = np.ones(len(np.asarray(list(URM_test.user_id)))*10)
        a = list(URM_test.user_id)
        a = [[element]*10 for element in a]

        b = list(URM_test.item_list)
        c = [el.split() for el in b]
        d = []
        for row in c:
            for el in row:
                d.append(int(el))
        d = np.asarray(d)
        a = np.asarray(a)
        a = a.flatten()
        URM_test = sps.coo_matrix((ratings_list, (a, d)))
        URM_test = URM_test.tocsr()
        return URM_test

    def convert_list_of_string_into_int(self, string_list):
        c = string_list[0].split()

        return [int(el) for el in c]


    def load_target_users_test(self):
        target_users_test_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users_test.csv"), "r")
        return list(target_users_test_file)[1:]

    def load_relevant_items(self):
        relevant_items = defaultdict(list)
        URM_test = self.load_URM_test()
        users_list_test = np.asarray(list(URM_test.user_id))
        items_list_test = np.asarray(list(URM_test.item_list))
        count = 0
        for user_id in users_list_test:
            relevant_items[user_id].append(items_list_test[count])
            count += 1
        return relevant_items

    def load_icm_asset(self):
        icm_asset = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_ICM_asset.csv"))
        row = np.asarray(list(icm_asset.row))
        col = np.asarray(list(icm_asset.col))
        data = np.asarray(list(icm_asset.data))
        icm_asset = sps.coo_matrix((data, (row, col)))
        icm_asset = icm_asset.tocsr()
        return icm_asset

    def load_icm_price(self):
        icm_price = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_ICM_price.csv"))
        row = np.asarray(list(icm_price.row))
        col = np.asarray(list(icm_price.col))
        data = np.asarray(list(icm_price.data))
        icm_price = sps.coo_matrix((data, (row, col)))
        icm_price = icm_price.tocsr()
        return icm_price

    def load_icm_sub_class(self):
        icm_sub_class = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_ICM_sub_class.csv"))
        row = np.asarray(list(icm_sub_class.row))
        col = np.asarray(list(icm_sub_class.col))
        data = np.asarray(list(icm_sub_class.data))
        icm_sub_class = sps.coo_matrix((data, (row, col)))
        icm_sub_class = icm_sub_class.tocsr()
        return icm_sub_class


    '''
    def load_tracks_matrix(self):
        tracks_matrix = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/tracks.csv"))
        return tracks_matrix

    def load_icm_album(self):
        tracks_matrix = self.load_tracks_matrix()
        track_ids = np.asarray(list(tracks_matrix.track_id))
        album_ids = np.asarray(list(tracks_matrix.album_id))
        ratings_list = np.ones(len(album_ids))
        icm_album = sps.coo_matrix((ratings_list, (track_ids, album_ids)))
        icm_album = icm_album.tocsr()
        return icm_album

    def load_icm_artist(self):
        tracks_matrix = self.load_tracks_matrix()
        track_ids = np.asarray(list(tracks_matrix.track_id))
        artist_ids = np.asarray(list(tracks_matrix.artist_id))
        ratings_list = np.ones(len(artist_ids))
        icm_artist = sps.coo_matrix((ratings_list, (track_ids, artist_ids)))
        icm_artist = icm_artist.tocsr()
        return icm_artist
    '''
    def bm25_normalization(self, matrix):
        matrix_BM25 = matrix.copy().astype(np.float32)
        matrix_BM25 = okapi_BM_25(matrix_BM25)
        matrix_BM25 = matrix_BM25.tocsr()
        return matrix_BM25
#e
    def tfidf_normalization(self, matrix):
        matrix_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(matrix)
        return matrix_tfidf.tocsr()

    def sklearn_normalization(self, matrix, axis=0):
        return normalize(matrix, axis=axis, norm='l2').tocsr()
