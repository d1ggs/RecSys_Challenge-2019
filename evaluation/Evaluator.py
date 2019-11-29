"""
Created on 11/11/19
@author: Matteo Carretta, Diego Piccinotti
"""
from random import randint
import os
from utils.helper import Helper
import numpy as np
from tqdm import tqdm

ROW_INDEX, COL_ID_INDEX = 0, 1
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBER_OF_ITEMS_TEST_FILE = 10


class Evaluator:
    def __init__(self, split_size=0.8):
        self.helper = Helper()
        _, self.test_data = self.helper.get_train_test_data()

    # Functions to evaluate a recommender
    def MAP(self, recommended_items, relevant_items):

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        # Cumulative sum: precision at 1, at 2, at 3 ...

        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score

    def evaluateRecommender(self, recommender):

        MAP_final = 0.0

        for user in tqdm(self.test_data.keys()):
            recommended_items = recommender.recommend(int(user), exclude_seen=True)[:10]
            relevant_item = self.test_data[int(user)]

            MAP_final += self.MAP(recommended_items, relevant_item)

        MAP_final /= len(self.test_data.keys())
        MAP_final *= 0.665
        result_string = "MAP@10 score: " + str(MAP_final)
        return MAP_final, result_string





