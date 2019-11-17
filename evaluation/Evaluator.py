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
        self.URM_test = pd.DataFrame(columns=["row", "col"])  # Matrix containing test data
        self.URM_train = self.helper.URM_data  # Matrix containing reduced training data, at the beginning it is equal to the test data (then it will be removed)
        self.target_users_test = None  # New list of playlists for which MAP will be computed
        self.split_size = split_size  # Percentage of training set that will be kept for training purposes (1-split_size is the size of the test set)
        self.dictionary_evaluation = None

    def split_data_randomly(self):
        # Instantiate an array containing the playlists to be put in new target_playlist_file
        users_list = np.asarray(list(self.URM_data.row))
        URM_csr = self.helper.convert_URM_to_csr(self.helper.URM_data)
        # Will contain target playlist ids of the test file
        self.target_users_test = np.empty(shape=users_list.shape)
        # Group by the URM just to have faster search
        grouped_by_user_URM = self.URM_data.groupby('row', as_index=True).apply(lambda x: list(x['col']))
        # Number of tuples of the test set, used for the upper bound over the counter
        test_dimensions = len(users_list) * (1 - self.split_size)

        # Instantiate csv file to write target playlists
        target_users_new_csv_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users_test.csv"), "w")
        target_users_new_csv_file.write("row\n")

        for count in tqdm(range(int(test_dimensions))):
            tracks_less_than_10 = True
            # Don't consider items with less than 10 elements (would not be good for testing since Kaggle evaluates with MAP10)
            while tracks_less_than_10:
                # Choose a random index between 0 and the length of URM's number of playlists
                random_index_to_be_put_in_test = randint(0, len(users_list)-1)
                candidate_user = users_list[random_index_to_be_put_in_test]
                # If there the playlist has fewer than 10 songs, we have to discard it
                if len(URM_csr[candidate_user].indices) >= NUMBER_OF_ITEMS_TEST_FILE:
                    # Append random playlist to playlist test set
                    self.target_users_test[count] = users_list[random_index_to_be_put_in_test]
                    # np array containing all the tracks contained in a selected playlist
                    item_list_10 = np.asarray(list(grouped_by_user_URM[candidate_user]))
                    # Keep just 10 tracks for each playlist
                    user_items_list = item_list_10[:NUMBER_OF_ITEMS_TEST_FILE]
                    # Create a tuple to be appended in URM_test matrix and append it
                    URM_test_tuple = pd.DataFrame({"row": [int(self.target_users_test[count])], "col": [str(user_items_list)]})
                    self.URM_test = self.URM_test.append(URM_test_tuple, ignore_index=True)

                    # Filter URM_train to eliminate test tuples
                    self.URM_train = self.URM_train.loc[~(self.URM_train["row"] == self.target_users_test[count])]

                    # Write target playlist for test set in its csv file
                    target_users_new_csv_file.write(str(int(self.target_users_test[count])) + "\n")

                    # Exit while loop
                    tracks_less_than_10 = False


        # Format data of test_data.csv correctly and save it
        self.URM_test["col"] = self.URM_test["col"].str.strip(' []')
        self.URM_test["col"] = self.URM_test["col"].replace('\s+', ' ', regex=True)
        self.URM_test.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/test_data.csv"), index=False)

        print("URM_test created, now creating URM_train")
        self.URM_train.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/train_data.csv"), index=False)


    # Functions to evaluate a recommender
    def MAP(recommended_items, relevant_items):

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score

    def evaluate(self, row, recommended_items):
        relevant_items = self.dictionary_evaluation[row]
        map = self.MAP(recommended_items, relevant_items)
        return map


