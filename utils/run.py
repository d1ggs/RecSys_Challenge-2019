from tqdm import tqdm

# from CollaborativeFilter import CollaborativeFilter
from SLIM.SLIMRecommender import SLIMRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from utils.helper import Helper
import os
from evaluation.Evaluator import Evaluator
import numpy as np

# Put root project dir in a constant
from utils.split_URM import split_train_test

ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RunRecommender:

    @staticmethod
    def run(recommender):

        # Helper contains methods to convert URM in CSR
        helper = Helper()
        URM_CSR = helper.convert_URM_to_csr(helper.URM_data)

        # Start recommendation
        recommender.fit(URM_CSR)

        RunRecommender.write_submission(recommender)

    @staticmethod
    def write_submission(recommender):
        submission_target = os.path.join(ROOT_PROJECT_PATH, "data/recommendation_matrix_to_submit.csv")
        print("Writing submission in", submission_target)

        # Open up target_users file
        target_users_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users.csv"), "r")
        target_users = list(target_users_file)[1:]  # Eliminate header

        # Instantiate a new submission file
        recommendation_matrix_file_to_submit = open(submission_target, 'w')
        recommendation_matrix_file_to_submit.write("user_id,item_list\n")
        for user in tqdm(target_users):
            items_recommended = recommender.recommend(int(user), exclude_seen=True)
            items_recommended = " ".join(str(x) for x in items_recommended)
            user = user.replace("\n", "")  # remove \n from user file
            recommendation_matrix_file_to_submit.write(user + "," + items_recommended + "\n")
        recommendation_matrix_file_to_submit.close()

    @staticmethod
    def perform_evaluation(recommender, test_data: dict):
        """Takes an already fitted recommender and evaluates on test data.
         If test_mode is false writes the submission"""

        if not test_data:
            print("Missing test data! Exiting...")
            exit(-1)

        print("Performing evaluation on test set...")

        MAP_final = 0.0
        evaluator = Evaluator()

        for user in tqdm(test_data.keys()):
            recommended_items = recommender.recommend(int(user), exclude_seen=True)
            relevant_item = test_data[int(user)]

            MAP_final += evaluator.MAP(recommended_items, relevant_item)

        MAP_final /= len(test_data.keys())

        print("MAP-10 score:", MAP_final)
        return MAP_final

