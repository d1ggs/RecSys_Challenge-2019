from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from sklearn import feature_extraction
import os

from utils.split_URM import split_train_test
from utils.data_utilities import build_URM

from Legacy.Base.IR_feature_weighting import okapi_BM_25
# Put root project dir in a global constant
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Helper:
    def __init__(self):
        # Put root project dir in a constant
        # Loading of data using pandas, that creates a new object URM_data with the first line values as attributes
        # (playlist_id, track_id) as formatted in the .csv file
        self.URM_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/dataset.csv"))
        self.URM_csr = self.convert_URM_to_csr(self.URM_data)
        self.users_list_data = np.asarray(list(self.URM_data.row))
        self.items_list_data = np.asarray(list(self.URM_data.col))
        self.URM_train = None
        self.test_data = None

    def get_train_test_data(self, resplit=False, split_fraction=0.8, leave_out=1):
        if self.URM_train is None or self.test_data is None:
            self.URM_train, _, self.test_data = split_train_test(self.URM_csr, split_fraction=split_fraction,
                                                                 rewrite=resplit, leave_out=leave_out)
        return self.URM_train.copy(), self.test_data.copy()

    def convert_URM_to_csr(self, URM):
        # TODO is it really necessary to convert to numpy array before calling ones?
        ratings_list = np.ones(len(np.asarray(list(URM.data))))

        # TODO implement data splitting here
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
        user_list = list(URM_test.user_id)

        ratings_list = list(URM_test.item_list)

        ratings_list = [el.split() for el in ratings_list]
        cols = []
        rows = []
        for i in range(len(ratings_list)):
            for _ in range(len(ratings_list[i])):
                rows.append(user_list[i])
            for el in ratings_list[i]:
                cols.append(int(el))

        cols = np.asarray(cols)
        rows = np.asarray(rows)
        rows = rows.flatten()

        print(rows.shape)
        print(cols.shape)
        ratings_list = np.ones(rows.shape[0])
        URM_test = sps.csr_matrix((ratings_list, (rows, cols)))
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

    def load_ucm_age(self):
        ucm_age = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_age.csv"))
        row = np.asarray(list(ucm_age.row))
        col = np.asarray(list(ucm_age.col))
        data = np.asarray(list(ucm_age.data))
        ucm_age = sps.coo_matrix((data, (row, col)))
        ucm_age = ucm_age.tocsr()
        return ucm_age

    def load_ucm_region(self):
        ucm_region = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region_no_zero_cols.csv"))
        row = np.asarray(list(ucm_region.row))
        col = np.asarray(list(ucm_region.col))
        data = np.asarray(list(ucm_region.data))
        ucm_region = sps.coo_matrix((data, (row, col)))
        ucm_region = ucm_region.tocsr()
        return ucm_region

    def split_ucm_region(self):
        ucm_region_file = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region.csv"))
        ucm_region_correct = ucm_region_file.loc[~(ucm_region_file.col == 0)]
        ucm_region_correct.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region_no_zero_cols.csv"), index=False)