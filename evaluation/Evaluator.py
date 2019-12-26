"""
Created on 11/11/19
@author: Matteo Carretta, Diego Piccinotti
"""
import copy
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

class Evaluator(object):
    def __init__(self, test_mode=False):
        self.helper = Helper()
        self.test_mode = test_mode
        self.evaluation_data = self.helper.test_data if self.test_mode else self.helper.validation_data

    def change_mode(self, mode):
        if mode == "test":
            self.test_mode = True
        if mode == "validation":
            self.test_mode = False
        else:
            raise AssertionError("Mode can be either test or validation, no other modes are supported")

        self.evaluation_data = self.helper.test_data if self.test_mode else self.helper.validation_data


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
        recommended_items = recommender.recommend(int(user), exclude_seen=True)
        return Evaluator.MAP(recommended_items, relevant_item)

    def evaluate_recommender_kfold(self, recommender, users_to_evaluate, evaluation_data, sequential=True):
        num_cores = multiprocessing.cpu_count()

        MAP_final = 0.0
        startTime = datetime.now()

        if sequential:
            print("Computing MAP sequentially...")
            for user in users_to_evaluate:
                recommended_items = recommender.recommend(int(user), exclude_seen=True)[:10]
                relevant_item = evaluation_data[int(user)]
                MAP_final += self.MAP(recommended_items, relevant_item)
        else:
            print("Computing MAP in parallel...")
            with multiprocessing.Pool(processes=num_cores) as p:
                results = p.map(Evaluator.compute_MAP_for_user,
                                [(recommender, user, evaluation_data[user]) for user in users_to_evaluate])
            MAP_final = np.sum(results)
            p.close()
        print("Completed in", datetime.now() - startTime, "\n")

        MAP_final /= len(users_to_evaluate)
        result_string = "MAP@10 score: " + str(MAP_final)
        return MAP_final, result_string

    def evaluate_recommender_on_cold_users(self, recommender, sequential=False):
        cold_users = Helper().get_cold_user_ids("test" if self.test_mode else "validation")
        cold_users = list(cold_users)

        ok_users = []
        for user in cold_users:
            if user in self.evaluation_data.keys():
                ok_users.append(user)

        cold_users = set(ok_users)

        return self.perform_evaluation(recommender, sequential, cold_users)


    def evaluateRecommender(self, recommender, users_to_evaluate: set, sequential=True):

        if users_to_evaluate is not None:
            user_list = users_to_evaluate
        else:
            user_list = self.evaluation_data.keys()

        MAP_final, result_string = self.perform_evaluation(recommender, sequential, user_list)

        return MAP_final, result_string

    def perform_evaluation(self, recommender, sequential, user_list):
        MAP_final = 0.0
        startTime = datetime.now()
        if sequential:
            print("Computing MAP sequentially...")
            for user in user_list:
                recommended_items = recommender.recommend(int(user), exclude_seen=True)[:10]
                relevant_item = self.evaluation_data[int(user)]
                MAP_final += self.MAP(recommended_items, relevant_item)
        else:
            print("Computing MAP in parallel...")
            num_cores = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_cores) as p:
                results = p.map(Evaluator.compute_MAP_for_user,
                                [(recommender, user, self.evaluation_data[user]) for user in user_list])
            MAP_final = np.sum(results)
            p.close()
        print("Completed in", datetime.now() - startTime, "\n")
        MAP_final /= len(user_list)
        result_string = "MAP@10 score: " + str(MAP_final)
        return MAP_final, result_string





