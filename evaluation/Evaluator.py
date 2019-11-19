"""
Created on 11/11/19
@author: Matteo Carretta
"""
from random import randint
import os
from utils.helper import Helper
import numpy as np
import pandas as pd
from tqdm import tqdm

ROW_INDEX, COL_ID_INDEX = 0, 1
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBER_OF_ITEMS_TEST_FILE = 10


class Evaluator:
    def __init__(self, split_size=0.8):
        self.helper = Helper()
        self.URM_data = self.helper.URM_data  # Matrix containing full original training data
        self.URM_test = pd.DataFrame(columns=["user_id", "item_list"])  # Matrix containing test data
        self.URM_train = self.helper.URM_data  # Matrix containing reduced training data, at the beginning it is equal to the test data (then it will be removed)
        self.target_users_test = None  # New list of playlists for which MAP will be computed
        self.split_size = split_size# Percentage of training set that will be kept for training purposes (1-split_size is the size of the test set)
        self.eval_targets = pd.DataFrame(columns=["user_id", "item_list"])
    def sampleTestUsers(self, URM_csr, sample_size):
        candidates = URM_csr.sum(1)
        indices = np.argwhere(candidates >= 11)[:, 0]
        if sample_size > indices.shape[0]:
            print("The sample requested is larger than the candidate users number. Returning whole vector.")
            return indices
        return np.random.choice(indices, sample_size, replace=False)

    def split_data_randomly(self):
        # Instantiate an array containing the users to be put in new target_users_file
        URM_csr = self.helper.convert_URM_to_csr(self.helper.URM_data)
        # Will contain target user ids of the test file
        # Group by the URM just to have faster search
        grouped_by_user_URM = self.URM_data.groupby('row', as_index=True).apply(lambda x: list(x['col']))

        # Number of tuples of the test set, used for the upper bound over the counter
        test_dimensions = URM_csr.shape[0] * (1 - self.split_size)
        random_indices = self.sampleTestUsers(URM_csr, int(test_dimensions))
        # Instantiate csv file to write target users
        target_users_new_csv_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users_test.csv"), "w")
        target_users_new_csv_file.write("user_id\n")

        for random_index in tqdm(random_indices):
            # Don't consider items with less than 10 elements (would not be good for testing since Kaggle evaluates with MAP10)
            # Choose a random index between 0 and the length of URM's number of playlists

            # Append random playlist to playlist test set
            # np array containing all the tracks contained in a selected playlist
            item_list_10 = np.asarray(list(grouped_by_user_URM[random_index]))

            # Keep just 10 tracks for each playlist
            user_items_list = item_list_10[:NUMBER_OF_ITEMS_TEST_FILE]
            # item_list_10 = self.helper.convert_list_of_string_into_int(user_items_list)

            # Create a tuple to be appended in URM_test matrix and append it
            URM_test_tuple = pd.DataFrame({"user_id": [int(random_index)], "item_list": [str(user_items_list)]})
            self.URM_test = self.URM_test.append(URM_test_tuple, ignore_index=True)

            # Filter URM_train to eliminate test tuples
            self.URM_train = self.URM_train.loc[~(self.URM_train["row"] == random_index)]

        self.target_users_test = np.asarray(list(self.URM_test.user_id))
        # Write target playlist for test set in its csv file
        for user in self.target_users_test:
            target_users_new_csv_file.write(str(int(user)) + "\n")


        # Format data of test_data.csv correctly and save it
        self.URM_test["item_list"] = self.URM_test["item_list"].str.strip(' []')
        self.URM_test["item_list"] = self.URM_test["item_list"].replace('\s+', ' ', regex=True)
        self.URM_test.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/test_data.csv"), index=False)

        print("URM_test created, now creating URM_train")
        self.URM_train.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/train_data.csv"), index=False)

    def split_data_randomly_2(self):
        # Instantiate an array containing the users to be put in new target_users_file
        URM_csr = self.helper.convert_URM_to_csr(self.helper.URM_data)
        # Will contain target user ids of the test file
        # Group by the URM just to have faster search
        grouped_by_user_URM = self.URM_data.groupby('row', as_index=True).apply(lambda x: list(x['col']))

        # Number of tuples of the test set, used for the upper bound over the counter
        test_dimensions = URM_csr.shape[0] * (1 - self.split_size)
        random_indices = self.sampleTestUsers(URM_csr, int(test_dimensions))
        # Instantiate csv file to write target users
        target_users_new_csv_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users_test.csv"), "w")
        target_users_new_csv_file.write("user_id\n")

        for random_index in tqdm(random_indices):
            # Don't consider items with less than 10 elements (would not be good for testing since Kaggle evaluates with MAP10)
            # Choose a random index between 0 and the length of URM's number of playlists

            # Append random playlist to playlist test set
            # np array containing all the tracks contained in a selected playlist
            item_list_10 = np.asarray(list(grouped_by_user_URM[random_index]))
            np.random.shuffle(item_list_10)

            evaluate_items = item_list_10[:NUMBER_OF_ITEMS_TEST_FILE]

            # Keep just 10 tracks for each playlist
            user_items_list = item_list_10[NUMBER_OF_ITEMS_TEST_FILE:]


            # Create a tuple to be appended in URM_test matrix and append it
            evaluate_tuple = pd.DataFrame({"user_id": [int(random_index)], "item_list": [str(evaluate_items)]})
            self.eval_targets = self.eval_targets.append(evaluate_tuple, ignore_index=True)

            # Create a tuple to be appended in URM_test matrix and append it
            URM_test_tuple = pd.DataFrame({"user_id": [int(random_index)], "item_list": [str(user_items_list)]})
            self.URM_test = self.URM_test.append(URM_test_tuple, ignore_index=True)

            # Filter URM_train to eliminate test tuples
            self.URM_train = self.URM_train.loc[~(self.URM_train["row"] == random_index)]

        self.target_users_test = np.asarray(list(self.URM_test.user_id))
        # Write target playlist for test set in its csv file
        for user in self.target_users_test:
            target_users_new_csv_file.write(str(int(user)) + "\n")

        # Format data of test_data.csv correctly and save it
        self.URM_test["item_list"] = self.URM_test["item_list"].str.strip(' []')
        self.URM_test["item_list"] = self.URM_test["item_list"].replace('\s+', ' ', regex=True)
        self.URM_test.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/test_data.csv"), index=False)

        # Format data of test_data.csv correctly and save it
        self.eval_targets["item_list"] = self.eval_targets["item_list"].str.strip(' []')
        self.eval_targets["item_list"] = self.eval_targets["item_list"].replace('\s+', ' ', regex=True)
        self.eval_targets.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/evaluate_data.csv"), index=False)

        print("URM_test created, now creating URM_train")
        self.URM_train.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/train_data.csv"), index=False)

    # Functions to evaluate a recommender
    def MAP(self, recommended_items, relevant_items):

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score


