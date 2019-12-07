import pickle
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

EVALUATION_DATA_PATH = ROOT_PROJECT_PATH + "/data/pickled/evaluation_data.pickle"
TEST_DATA_PATH = ROOT_PROJECT_PATH + "/data/pickled/test_data.pickle"
URM_TRAIN_EVAL_ONLY_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_train_eval_only.pickle"
URM_TRAIN_TEST_EVAL_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_train_eval_test.pickle"
URM_TEST_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_test.pickle"


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def compute_cold_user_ids(URM):

    interactions = URM.sum(1)

    # Get the users with no interactions
    indices = np.argwhere(interactions == 0)[:, 0]
    return set(indices)


class Helper(object, metaclass=Singleton):
    def __init__(self, resplit=False):
        # Put root project dir in a constant
        # Loading of data using pandas, that creates a new object URM_data with the first line values as attributes
        # (playlist_id, track_id) as formatted in the .csv file
        self.URM_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/dataset.csv"))
        self.URM_csr = self.convert_URM_to_csr(self.URM_data)
        self.users_list_data = np.asarray(list(self.URM_data.row))
        self.items_list_data = np.asarray(list(self.URM_data.col))

        self.URM_train_validation, self.URM_train_test, self.validation_data, self.test_data = self.get_train_validation_test_data(
            resplit=resplit)

        self.cold_users_dataset = compute_cold_user_ids(self.URM_csr)
        self.cold_users_validation = compute_cold_user_ids(self.URM_train_validation)
        self.cold_users_test = compute_cold_user_ids(self.URM_train_test)

    def get_number_of_users(self):
        return self.URM_csr.shape[0]

    def get_cold_user_ids(self, split: str):
        if split == "dataset":
            return self.cold_users_dataset
        elif split == "test":
            return self.cold_users_test
        elif split == "validation":
            return self.cold_users_validation
        else:
            raise AssertionError("The split parameter must be either 'dataset', 'test', or 'validation'")

    def get_train_validation_test_data(self, resplit=False, split_fraction=0, leave_out=1):

        if not resplit:
            # Load serialized Pickle data
            print("Loading pickled data")

            evaluation_data_in = open(EVALUATION_DATA_PATH, "rb")
            eval_data = pickle.load(evaluation_data_in)
            evaluation_data_in.close()

            URM_train_in = open(URM_TRAIN_EVAL_ONLY_PATH, "rb")
            URM_train_eval = pickle.load(URM_train_in)
            URM_train_in.close()

            test_data_in = open(TEST_DATA_PATH, "rb")
            test_data = pickle.load(test_data_in)
            test_data_in.close()

            URM_train_in = open(URM_TRAIN_TEST_EVAL_PATH, "rb")
            URM_train_test = pickle.load(URM_train_in)
            URM_train_in.close()

            # URM_test_in = open(URM_TEST_PATH, "rb")
            # self.URM_test = pickle.load(URM_test_in)
            # URM_test_in.close()

        else:
            URM_train_eval, URM_train_test, eval_data, test_data = split_train_test(self.URM_csr,
                                                                                    split_fraction=split_fraction,
                                                                                    leave_out=leave_out)

            # Serialize the objects with Pickle, for faster loading

            print("Saving Pickle files into /data/pickled")

            test_data_out = open(TEST_DATA_PATH, "wb")
            pickle.dump(test_data, test_data_out)
            test_data_out.close()

            URM_train_out = open(URM_TRAIN_TEST_EVAL_PATH, "wb")
            pickle.dump(URM_train_test, URM_train_out)
            URM_train_out.close()

            eval_data_out = open(EVALUATION_DATA_PATH, "wb")
            pickle.dump(eval_data, eval_data_out)
            eval_data_out.close()

            URM_train_out = open(URM_TRAIN_EVAL_ONLY_PATH, "wb")
            pickle.dump(URM_train_eval, URM_train_out)
            URM_train_out.close()

            # URM_test_out = open(URM_TEST_PATH, "wb")
            # pickle.dump(URM_test, URM_test_out)
            # URM_test_out.close()

        return URM_train_eval, URM_train_test, eval_data, test_data

    def convert_URM_to_csr(self, URM):
        ratings_list = np.ones(len(np.asarray(list(URM.data))))
        URM = sps.coo_matrix((ratings_list, (np.asarray(list(URM.row)), np.asarray(list(URM.col)))))
        URM = URM.tocsr()
        return URM

    def load_target_users_test(self):
        target_users_test_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users_test.csv"), "r")
        return list(target_users_test_file)[1:]

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

    # e
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
        ucm_age = sps.coo_matrix((data, (row, col)), shape=(self.URM_csr.shape[0], max(col) + 1))
        ucm_age = ucm_age.tocsr()
        return ucm_age

    def load_ucm_region(self):
        ucm_region = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region.csv"))
        row = np.asarray(list(ucm_region.row))
        col = np.asarray(list(ucm_region.col))
        data = np.asarray(list(ucm_region.data))
        ucm_region = sps.coo_matrix((data, (row, col)), shape=(self.URM_csr.shape[0], max(col) + 1))
        ucm_region = ucm_region.tocsr()
        return ucm_region

    def split_ucm_region(self):
        ucm_region_file = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region.csv"))
        ucm_region_correct = ucm_region_file.loc[~(ucm_region_file.col == 0)]
        ucm_region_correct.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region_no_zero_cols.csv"), index=False)
