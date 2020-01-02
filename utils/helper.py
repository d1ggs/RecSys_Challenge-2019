import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from sklearn import feature_extraction
import os
from math import ceil

from utils.split_URM import split_train_test

from Legacy.Base.IR_feature_weighting import okapi_BM_25

# Put root project dir in a global constant
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVALUATION_DATA_PATH = ROOT_PROJECT_PATH + "/data/pickled/evaluation_data.pickle"
TEST_DATA_PATH = ROOT_PROJECT_PATH + "/data/pickled/test_data.pickle"
URM_TRAIN_EVAL_ONLY_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_train_eval_only.pickle"
URM_TRAIN_TEST_EVAL_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_train_eval_test.pickle"
URM_TEST_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_test.pickle"


class DoubleStructure(object):
    def __init__(self, age=None, region=None, cluster=None):
        self.age = age
        self.region = region
        self.cluster = cluster


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            print("Instantiating new singleton")
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Helper(object, metaclass=Singleton):
    def __init__(self, resplit=False):
        self.URM_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/dataset.csv"))
        self.URM_csr = self.convert_URM_to_csr(self.URM_data)
        self.users_list_data = np.asarray(list(self.URM_data.row))
        self.items_list_data = np.asarray(list(self.URM_data.col))

        self.kfold_times = 1
        self.kfold_split = None
        self.URM_train_validation, self.URM_train_test, self.validation_data, self.test_data = self.get_train_validation_test_data(
            resplit=resplit)

        self.cold_users_dataset, self.warm_users_dataset = self.compute_cold_warm_user_ids(self.URM_csr)
        self.cold_users_validation, self.warm_users_validation = self.compute_cold_warm_user_ids(self.URM_train_validation)
        self.cold_users_test, self.warm_users_test = self.compute_cold_warm_user_ids(self.URM_train_test)

    def compute_cold_warm_user_ids(self, URM):
        interactions = URM.sum(1)

        # Get the users with no interactions
        cold_indices = np.argwhere(interactions == 0)[:, 0]
        warm_indices = np.argwhere(interactions > 0)[:, 0]
        return set(cold_indices), set(warm_indices)

    @staticmethod
    def load_predictions(file_path):
        return pd.read_csv(ROOT_PROJECT_PATH + file_path)

    @staticmethod
    def load_user_df(file_path="/data/dataframes/user.csv"):
        return pd.read_csv(ROOT_PROJECT_PATH + file_path)

    @staticmethod
    def load_item_df(file_path="/data/dataframes/item.csv"):
        return pd.read_csv(ROOT_PROJECT_PATH + file_path)

    @staticmethod
    def build_user_dataframe(file_path="/data/dataframes/user.csv"):
        user_region = pd.read_csv(ROOT_PROJECT_PATH + "data/data_UCM_region.csv")
        user_age = pd.read_csv(ROOT_PROJECT_PATH + "data/data_UCM_age.csv")

        user_region.drop(columns=["data"])
        user_region.rename(columns={"col": "region", "row": "user_id"}, inplace=True)
        user_age.drop(columns=["data"])
        user_age.rename(columns={"col": "age", "row": "user_id"}, inplace=True)

        user_df = pd.merge(user_age, user_region, on="user_id", how='outer')
        user_df.to_csv(path_or_buf=ROOT_PROJECT_PATH + file_path)

    @staticmethod
    def build_item_dataframe(file_path="/data/dataframes/item.csv"):
        item_price = pd.read_csv(ROOT_PROJECT_PATH + "data/data_ICM_price.csv")
        item_asset = pd.read_csv(ROOT_PROJECT_PATH + "data/data_ICM_asset.csv")
        item_sub_class = pd.read_csv(ROOT_PROJECT_PATH + "data/data_ICM_sub_class.csv")

        data = np.array(list(item_price.col))

        le = LabelEncoder()
        le.fit(data)
        data = le.transform(data)

        data_list = []
        for el in data:
            data_list.append(el)

        item_price.drop(columns=["data"])
        item_price.insert(1, "price", data_list, True)

        data = np.array(list(item_asset.col))

        le = LabelEncoder()
        le.fit(data)
        data = le.transform(data)

        data_list = []
        for el in data:
            data_list.append(el)

        item_asset.drop(columns=["data"])
        item_asset.insert(1, "asset", data_list, True)

        item_sub_class.drop(columns=["data"])
        item_sub_class.rename(columns={"col": "sub_class", "row": "item_id"}, inplace=True)
        item_asset.drop(columns=["col"])
        item_asset.rename(columns={"row": "item_id"}, inplace=True)
        item_price.drop(columns=["col"])
        item_asset.rename(columns={"row": "item_id"}, inplace=True)

        user_df = pd.merge(item_price, item_asset, item_sub_class, on="item_id", how='outer')
        user_df.to_csv(path_or_buf=ROOT_PROJECT_PATH + file_path)


    def get_number_of_users(self):
        return self.URM_csr.shape[0]

    def get_kfold_data(self, k):
        if not (k == self.kfold_times and self.kfold_split):
            self.kfold_times = k
            self.kfold_split = []
            for _ in range(k):
                self.kfold_split.append(self.get_train_validation_test_data(resplit=True, save_pickle=False))
        return self.kfold_split

    def get_cold_user_ids(self, split: str):
        if split == "dataset":
            return self.cold_users_dataset
        elif split == "test":
            return self.cold_users_test
        elif split == "validation":
            return self.cold_users_validation
        else:
            raise AssertionError("The split parameter must be either 'dataset', 'test', or 'validation'")

    def get_warm_user_ids(self, split: str):
        if split == "dataset":
            return self.warm_users_dataset
        elif split == "test":
            return self.warm_users_test
        elif split == "validation":
            return self.warm_users_validation
        else:
            raise AssertionError("The split parameter must be either 'dataset', 'test', or 'validation'")

    def get_train_validation_test_data(self, resplit=False, save_pickle=True, split_fraction=0):

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
                                                                                    split_fraction=split_fraction)

            # Serialize the objects with Pickle, for faster loading

            if save_pickle:
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

    def load_icm_asset(self, cluster=False):
        le = LabelEncoder()
        icm_asset = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_ICM_asset.csv"))
        row = np.asarray(list(icm_asset.row))
        col = np.asarray(list(icm_asset.data))
        data = np.ones_like(col)

        le.fit(col)
        col = le.transform(col)

        if cluster:
            n_clusters = 10
            max_value = np.max(data)
            cluster_size = max_value / n_clusters
            cols = []
            new_data = []
            for value in data:
                cols.append(ceil(value / cluster_size) - 1)
                new_data.append(1)
            col = cols
            data = new_data

        icm_asset = sps.coo_matrix((data, (row, col)))
        icm_asset = icm_asset.tocsr()
        return icm_asset

    def load_icm_price(self, cluster=False):
        le = LabelEncoder()
        icm_price = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_ICM_price.csv"))
        row = np.asarray(list(icm_price.row))
        col = np.asarray(list(icm_price.data))
        data = np.ones_like(col)

        le.fit(col)
        col = le.transform(col)

        if cluster:
            n_clusters = 10
            max_value = np.max(data)
            cluster_size = max_value / n_clusters
            cols = []
            new_data = []
            for value in data:
                cols.append(ceil(value / cluster_size) - 1)
                new_data.append(1)
            col = cols
            data = new_data
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
