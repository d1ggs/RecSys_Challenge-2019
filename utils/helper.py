import pickle
from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from sklearn import feature_extraction
from sklearn.cluster import KMeans
import os
from math import ceil

from utils.split_URM import split_train_test

from Legacy.Base.IR_feature_weighting import okapi_BM_25
from tqdm import tqdm

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
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def compute_cold_warm_user_ids(URM):
    interactions = URM.sum(1)

    # Get the users with no interactions
    cold_indices = np.argwhere(interactions == 0)[:, 0]
    warm_indices = np.argwhere(interactions > 0)[:, 0]
    return set(cold_indices), set(warm_indices)


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

        self.cold_users_dataset, self.warm_users_dataset = compute_cold_warm_user_ids(self.URM_csr)
        self.cold_users_validation, self.warm_users_validation = compute_cold_warm_user_ids(self.URM_train_validation)
        self.cold_users_test, self.warm_users_test = compute_cold_warm_user_ids(self.URM_train_test)

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

    def get_warm_user_ids(self, split: str):
        if split == "dataset":
            return self.warm_users_dataset
        elif split == "test":
            return self.warm_users_test
        elif split == "validation":
            return self.warm_users_validation
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

    def load_icm_asset(self, cluster=True):
        icm_asset = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_ICM_asset.csv"))
        row = np.asarray(list(icm_asset.row))
        col = np.asarray(list(icm_asset.col))
        data = np.asarray(list(icm_asset.data))

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

    def load_icm_price(self, cluster=True):
        icm_price = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_ICM_price.csv"))
        row = np.asarray(list(icm_price.row))
        col = np.asarray(list(icm_price.col))
        data = np.asarray(list(icm_price.data))

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

    #
    # def tail_boost(self, URM, step=1, lastN=4, mode="validation"):
    #     if mode == "validation":
    #         target_users = self.validation_data.keys()
    #     elif mode == "test":
    #         target_users = self.test_data.keys()
    #     elif mode == "dataset":
    #         target_users = self.users_list_data
    #     else:
    #         raise AssertionError("Mode selected not supported for tail boosting, choose between validation, test, dataset")
    #     target_users = set(target_users)
    #     for row_index in tqdm(range(URM.shape[0])):
    #         if row_index in target_users:
    #             items = URM[row_index].indices
    #             # sorted_items = np.intersect1d(sorted_items, items)
    #             len_items = len(items)
    #
    #             for i in range(len_items):
    #                 # THE COMMA IS IMPORTANT
    #
    #                 if len_items - i <= lastN:
    #                     additive_score = ((lastN + 1) - (len_items - i)) * step
    #                     URM.data[URM.indptr[row_index] + i] += additive_score
    #     return URM

    def get_demographic_info(self, verbose=False, mode="dataset", num_cluster=6):
        ucm_age = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_age.csv"))
        ucm_region = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region.csv"))

        # Load the cold users
        cold_users = np.array(list(self.get_cold_user_ids(mode)))
        if verbose:
            print("Total user number:", self.URM_csr.shape[0])
            print("Cold users number:", cold_users.shape[0])

        # Load the indices and data of users which have some demographic information
        row_age = np.asarray(list(ucm_age.row))
        data_age = np.asarray(list(ucm_age.col))

        if verbose:
            print("Users for which age data is available:", row_age.shape[0])

        row_region = np.asarray(list(ucm_region.row))
        data_region = np.asarray(list(ucm_region.col))
        if verbose:
            print("Users for which region data is available:", row_region.shape[0])

        # Compute the array of indices of the whole dataset
        user_ids = np.arange(0, self.URM_csr.shape[0])
        mask = np.isin(user_ids, cold_users, assume_unique=True, invert=True)
        warm_users = user_ids[mask]

        # Find users which miss age data
        mask = np.isin(user_ids, row_age, invert=True, assume_unique=True)
        missing_age = user_ids[mask]

        # Of these, check which are also cold users
        mask = np.isin(missing_age, cold_users, invert=False, assume_unique=True)
        missing_age_and_cold = missing_age[mask]
        missing_age_and_warm = missing_age[np.invert(mask)]

        # Find which cold users have age data instead
        mask = np.isin(cold_users, missing_age, invert=True, assume_unique=True)
        cold_and_age_available = cold_users[mask]
        mask = np.isin(warm_users, missing_age, invert=True, assume_unique=True)
        warm_and_age_available = warm_users[mask]

        if verbose:
            print("There are", missing_age_and_cold.shape[0], "cold users for which age data is not available")
            print("There are", cold_and_age_available.shape[0], "cold users for which age data is available")

        # Find users which miss region data
        mask = np.isin(user_ids, np.unique(row_region), invert=True, assume_unique=True)
        missing_region = user_ids[mask]

        # Of these, check which are also cold users
        mask = np.isin(missing_region, cold_users, invert=False, assume_unique=True)
        missing_region_and_cold = missing_region[mask]
        missing_region_and_warm = missing_region[np.invert(mask)]

        # Find which cold users have region data instead
        mask = np.isin(cold_users, missing_region, invert=True, assume_unique=True)
        cold_and_region_available = cold_users[mask]
        mask = np.isin(warm_users, missing_region, invert=True, assume_unique=True)
        warm_and_region_available = warm_users[mask]

        if verbose:
            print("There are", missing_region_and_cold.shape[0], "cold users for which region data is not available")
            print("There are", cold_and_region_available.shape[0], "cold users for which region data is available")

        # double_data_users = np.intersect1d(row_age, row_region, assume_unique=False)

        # Find the cold users which have both kind of demographic data
        double_data_cold_users = np.intersect1d(cold_and_age_available, cold_and_region_available, assume_unique=True)

        if verbose:
            print("There are", double_data_cold_users.shape[0],
                  "cold users for which both age and region data is available,"
                  "use KMeans clustering data to fill URM")

        # Compute the cold users which hold no kind of demographic data
        mask = np.isin(cold_users, cold_and_age_available, invert=True, assume_unique=True)
        cold_no_age = cold_users[mask]
        mask = np.isin(cold_no_age, cold_and_region_available, invert=True, assume_unique=True)
        completely_cold = cold_no_age[mask]

        if verbose:
            print("There are", completely_cold.shape[0],
                  "cold users for which neither age or region data is available, "
                  "resort to TopPop for these")

        # Compute cold users with just one of the two types of demographic data

        cold_region_only_users = np.intersect1d(cold_and_region_available, missing_age_and_cold)
        cold_age_only_users = np.intersect1d(cold_and_age_available, missing_region_and_cold)
        warm_region_only_users = np.intersect1d(warm_and_region_available, missing_age_and_warm)
        warm_age_only_users = np.intersect1d(warm_and_age_available, missing_region_and_warm)

        if verbose:
            print("There are", cold_age_only_users.shape[0], "cold users for which age data is available, "
                                                             "but no region data has been found")
            print("There are", cold_region_only_users.shape[0], "cold users for which region data is available, "
                                                                "but no region age has been found")

        # Compute users that have both kind of demographic information
        double_data = np.intersect1d(np.unique(row_region), row_age, assume_unique=True)

        # Extract region and age values for these users
        region_indices_double = []
        age_indices_double = []
        for user in double_data:
            region_indices_double.append(np.where(row_region == user)[0][0])
            age_indices_double.append(np.where(row_region == user)[0])

        region_indices_double = np.array(region_indices_double).squeeze()
        age_indices_double = np.array(age_indices_double).squeeze()

        region_data_double = data_region[region_indices_double].squeeze()
        age_data_double = data_age[age_indices_double].squeeze()

        double_data_dict = {}
        kmeans_data = []

        # Put them together to perform KMeans
        for user, age, region in zip(double_data, age_data_double, region_data_double):
            double_data_dict[user] = DoubleStructure(age=age, region=region)
            kmeans_data.append([region, age])

        kmeans_data = np.array(kmeans_data)

        # wcss = []
        # for i in range(1, 10):
        #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        #     kmeans.fit(kmeans_data)
        #     wcss.append(kmeans.inertia_)
        # plt.plot(range(1, 10), wcss)
        # plt.title('Elbow Method')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('WCSS')
        # plt.show()

        kmeans = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
        # kmeans.fit(kmeans_data)
        kmeans.fit(kmeans_data)

        # Divide users with both kind of demographic data into cold and warm users

        warm_mask = np.isin(double_data, cold_users, invert=True, assume_unique=True)
        cold_mask = np.invert(warm_mask)
        double_data_warm_users = double_data[warm_mask]
        double_data_cold_users = double_data[cold_mask]

        # Retrieve the associated data

        age_warm = age_data_double[warm_mask]
        age_cold = age_data_double[cold_mask]
        region_warm = region_data_double[warm_mask]
        region_cold = region_data_double[cold_mask]

        # Compute KMeans clusters

        kmeans_warm = []
        for user, age, region in zip(double_data_warm_users, age_warm, region_warm):
            # double_data_dict[user] = DoubleStructure(age=age, region=region)
            kmeans_warm.append([region, age])

        kmeans_cold = []
        for user, age, region in zip(double_data_cold_users, age_cold, region_cold):
            # double_data_dict[user] = DoubleStructure(age=age, region=region)
            kmeans_cold.append([region, age])

        pred_warm = kmeans.predict(np.array(kmeans_warm))
        pred_cold = kmeans.predict(np.array(kmeans_cold))

        # Compute data for users with only one type of data

        cold_region_indices = []
        cold_age_indices = []
        warm_region_indices = []
        warm_age_indices = []

        for user in cold_region_only_users:
            indices = np.where(row_region == user)[0]
            cold_region_indices.append(indices[0])

        for user in cold_age_only_users:
            cold_age_indices.append(np.where(row_age == user))

        cold_region_indices = np.array(cold_region_indices).squeeze()
        cold_age_indices = np.array(cold_age_indices).squeeze()

        cold_region_only_data = data_region[cold_region_indices].squeeze()
        cold_age_only_data = data_age[cold_age_indices].squeeze()

        for user in warm_region_only_users:
            indices = np.where(row_region == user)[0]
            warm_region_indices.append(indices[0])

        for user in warm_age_only_users:
            warm_age_indices.append(np.where(row_age == user))

        warm_region_indices = np.array(cold_region_indices).squeeze()
        warm_age_indices = np.array(cold_age_indices).squeeze()

        warm_region_only_data = data_region[warm_region_indices].squeeze()
        warm_age_only_data = data_age[warm_age_indices].squeeze()

        # Put info in the dictionaries, to retrieve it quickly

        user_data_dict = {}
        cluster_dict = {} # Contains cluster of each user
        region_dict = {}
        age_dict = {}

        for user, cluster in zip(double_data_warm_users, pred_warm):
            # user_data_dict[user] = DoubleStructure(cluster=cluster)
            if cluster in cluster_dict:
                cluster_dict[cluster].append(user)
            else:
                cluster_dict[cluster] = [user]

        for user, cluster in zip(double_data_cold_users, pred_cold):
            user_data_dict[user] = DoubleStructure(cluster=cluster)

        for user, region in zip(warm_region_only_users, warm_region_only_data):
            # user_data_dict[user] = DoubleStructure(region=region)
            if region in region_dict:
                region_dict[region].append(user)
            else:
                region_dict[region] = [user]

        for user, region in zip(cold_region_only_users, cold_region_only_data):
            user_data_dict[user] = DoubleStructure(region=region)

        for user, age in zip(warm_age_only_users, warm_age_only_data):
            # user_data_dict[user] = DoubleStructure(age=age)
            if age in age_dict:
                age_dict[age].append(user)
            else:
                age_dict[age] = [user]

        for user, age in zip(cold_age_only_users, cold_age_only_data):
            user_data_dict[user] = DoubleStructure(age=age)

        return user_data_dict, cluster_dict, region_dict, age_dict

    def inject_demographic_info(self, URM, verbose=False):
        raise NotImplementedError("The method must still be completed")

        ucm_age = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_age.csv"))
        ucm_region = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/data_UCM_region.csv"))

        # Load the cold users
        cold_users = np.array(list(self.get_cold_user_ids("dataset")))
        if verbose:
            print("Total user number:", URM.shape[0])
            print("Cold users number:", cold_users.shape[0])

        # Load the indices and data of users which have some demographic information
        row_age = np.asarray(list(ucm_age.row))
        data_age = np.asarray(list(ucm_age.col))

        if verbose:
            print("Users for which age data is available:", row_age.shape[0])

        row_region = np.asarray(list(ucm_region.row))
        data_region = np.asarray(list(ucm_region.col))
        if verbose:
            print("Users for which region data is available:", row_region.shape[0])

        # Compute the array of indices of the whole dataset
        user_ids = np.arange(0, URM.shape[0])

        # Find users which miss age data
        mask = np.isin(user_ids, row_age, invert=True)
        missing_age = user_ids[mask]

        # Of these, check which are also cold users
        mask = np.isin(cold_users, missing_age, invert=False)
        missing_age_and_cold = cold_users[mask]

        # Find which cold users have age data instead
        cold_and_age_available = cold_users[np.invert(mask)]

        if verbose:
            print("There are", missing_age_and_cold.shape[0], "cold users for which age data is not available")
            print("There are", cold_and_age_available.shape[0], "cold users for which age data is available")

        # Find users which miss region data
        mask = np.isin(user_ids, np.unique(row_region), invert=True)
        missing_region = user_ids[mask]

        # Of these, check which are also cold users
        mask = np.isin(cold_users, missing_region, invert=False)
        missing_region_and_cold = cold_users[mask]

        # Find which cold users have region data instead
        cold_and_region_available = cold_users[np.invert(mask)]

        if verbose:
            print("There are", missing_region_and_cold.shape[0], "cold users for which region data is not available")
            print("There are", cold_and_region_available.shape[0], "cold users for which region data is available")

        # double_data_users = np.intersect1d(row_age, row_region, assume_unique=False)

        # Find the cold users which have both kind of demographic data
        double_data_cold_users = np.intersect1d(cold_and_age_available, cold_and_region_available, assume_unique=True)

        if verbose:
            print("There are", double_data_cold_users.shape[0],
                  "cold users for which both age and region data is available,"
                  "use KMeans clustering data to fill URM")

        # Compute the cold users which hold no kind of demographic data
        mask = np.isin(cold_users, cold_and_age_available, invert=True)
        cold_no_age = cold_users[mask]
        mask = np.isin(cold_no_age, cold_and_region_available, invert=True)
        completely_cold = cold_no_age[mask]

        if verbose:
            print("There are", completely_cold.shape[0],
                  "cold users for which neither age or region data is available, "
                  "resort to TopPop for these")

        # Compute cold users with just one of the two types of demographic data

        cold_yes_region_no_age = np.intersect1d(cold_and_region_available, missing_age_and_cold)
        cold_yes_age_no_region = np.intersect1d(cold_and_age_available, missing_region_and_cold)

        if verbose:
            print("There are", cold_yes_age_no_region.shape[0], "cold users for which age data is available, "
                                                                "but no region data has been found")
            print("There are", cold_yes_region_no_age.shape[0], "cold users for which region data is available, "
                                                                "but no region age has been found")

        double_data = np.intersect1d(np.unique(row_region), row_age, assume_unique=True)

        region_indices = []
        age_indices = []
        for user in double_data:
            region_indices.append(np.where(row_region == user)[0][0])
            age_indices.append(np.where(row_region == user)[0])

        region_indices = np.array(region_indices).squeeze()
        age_indices = np.array(age_indices).squeeze()

        region_data = data_region[region_indices].squeeze()
        age_data = data_age[age_indices].squeeze()

        double_data_dict = {}
        kmeans_data = []

        for user, age, region in zip(double_data, age_data, region_data):
            double_data_dict[user] = DoubleStructure(age=age, region=region)
            kmeans_data.append([region, age])

        kmeans_data = np.array(kmeans_data)

        # wcss = []
        # for i in range(1, 10):
        #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        #     kmeans.fit(kmeans_data)
        #     wcss.append(kmeans.inertia_)
        # plt.plot(range(1, 10), wcss)
        # plt.title('Elbow Method')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('WCSS')
        # plt.show()

        mask = np.isin(double_data, cold_users, invert=True)
        double_data_warm_users = double_data[mask]

        kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
        # kmeans.fit(kmeans_data)
        kmeans.fit(kmeans_data)

        cluster_dict = {}
        URM = URM.copy()

        # TODO kmeans and mean only on WARM users
        region_indices = []
        age_indices = []
        for user in double_data:
            region_indices.append(np.where(row_region == user)[0][0])
            age_indices.append(np.where(row_region == user)[0])

        region_indices = np.array(region_indices).squeeze()
        age_indices = np.array(age_indices).squeeze()

        region_data = data_region[region_indices].squeeze()
        age_data = data_age[age_indices].squeeze()

        kmeans_data = []

        for user, age, region in zip(double_data, age_data, region_data):
            double_data_dict[user] = DoubleStructure(age=age, region=region)
            kmeans_data.append([region, age])

        kmeans_data = np.array(kmeans_data)

        pred_y = kmeans.predict(kmeans_data)

        # TODO do it for age only and region only

        for user, cluster in zip(double_data_warm_users, pred_y):
            double_data_dict[user].cluster = cluster
            if cluster in cluster_dict:
                cluster_dict[cluster].append(URM[user])
            else:
                cluster_dict[cluster] = [URM[user]]

        for cluster in cluster_dict.keys():
            row_sum = np.zeros_like(URM[1])
            members = 0
            for row in cluster_dict[cluster]:
                row_sum += row
                members += 1
            cluster_dict[cluster] = row_sum / members

        URM = sps.lil_matrix(URM)

        for user in double_data_cold_users:
            cluster = double_data_dict[user].cluster
            URM[user] = cluster_dict[cluster]

        URM = URM.tocsr()

        region_indices = []
        age_indices = []

        for user in cold_yes_region_no_age:
            indices = np.where(row_region == user)[0]
            region_indices.append(indices[0])

        for user in cold_yes_age_no_region:
            age_indices.append(np.where(row_age == user))

        region_indices = np.array(region_indices).squeeze()
        age_indices = np.array(age_indices).squeeze()

        region_data = data_region[region_indices].squeeze()
        age_data = data_age[age_indices].squeeze()

        regions_dict = {}
        age_dict = {}

        for user, region in zip(row_region, data_region):
            if region in regions_dict:
                regions_dict[region].append(URM[user])
            else:
                regions_dict[region] = [URM[user]]

        for user, age in zip(row_age, data_age):
            if age in age_dict:
                age_dict[age].append(URM[user])
            else:
                age_dict[age] = [URM[user]]

        for age in age_dict.keys():
            row_sum = np.zeros_like(URM[1])
            members = 0
            for row in age_dict[age]:
                row_sum += row
                members += 1
            age_dict[age] = row_sum / members

        for region in regions_dict.keys():
            row_sum = np.zeros_like(URM[1])
            members = 0
            for row in regions_dict[region]:
                row_sum += row
                members += 1
            regions_dict[region] = row_sum / members

        URM = sps.lil_matrix(URM)

        for user, region in zip(cold_yes_region_no_age, region_data):
            URM[user] = regions_dict[region]

        for user, age in zip(cold_yes_age_no_region, age_data):
            URM[user] = age_dict[age]

        return URM.tocsr()
