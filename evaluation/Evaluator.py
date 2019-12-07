"""
Created on 11/11/19
@author: Matteo Carretta, Diego Piccinotti
"""
from _datetime import datetime
from random import randint
import os
from utils.helper import Helper
import numpy as np
from tqdm import tqdm
import multiprocessing

ROW_INDEX, COL_ID_INDEX = 0, 1
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBER_OF_ITEMS_TEST_FILE = 10


class Evaluator:
    def __init__(self, split_size=0.8, test_mode = False):
        self.helper = Helper()
        self.test_mode = test_mode

    # Functions to evaluate a recommender
    @staticmethod
    def MAP(recommended_items, relevant_items):

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        # Cumulative sum: precision at 1, at 2, at 3 ...

        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score

    @staticmethod
    def compute_MAP_for_user(params):
        recommender = params[0]
        user = params[1]
        relevant_item = params[2]
        recommended_items = recommender.recommend(int(user), exclude_seen=True)[:10]
        return Evaluator.MAP(recommended_items, relevant_item)

    def evaluateRecommender_old(self, recommender, exclude_users: set):

        print("Computing MAP...")

        MAP_final = 0.0

        if self.test_mode:
            evaluation_data = self.helper.test_data
        else:
            evaluation_data = self.helper.validation_data

        if exclude_users is not None:
            user_list = exclude_users
        else:
            user_list = evaluation_data.keys()

        for user in tqdm(user_list):
            recommended_items = recommender.recommend(int(user), exclude_seen=True)[:10]
            relevant_item = evaluation_data[int(user)]
            MAP_final += self.MAP(recommended_items, relevant_item)

        MAP_final /= len(user_list)
        result_string = "MAP@10 score: " + str(MAP_final)
        return MAP_final, result_string

    def evaluateRecommender(self, recommender, exclude_users: set):

        num_cores = multiprocessing.cpu_count()

        print("Computing MAP...")

        MAP_final = 0.0

        if self.test_mode:
            evaluation_data = self.helper.test_data
        else:
            evaluation_data = self.helper.validation_data

        if exclude_users is not None:
            user_list = exclude_users
        else:
            user_list = evaluation_data.keys()

        startTime = datetime.now()
        with multiprocessing.Pool(processes=num_cores) as p:
            results = p.map(Evaluator.compute_MAP_for_user, [(recommender, user, evaluation_data[user]) for user in user_list])
        MAP_final = np.sum(results)
        p.close()

        print("Completed in", datetime.now() - startTime)

        MAP_final /= len(user_list)
        result_string = "MAP@10 score: " + str(MAP_final)
        return MAP_final, result_string





