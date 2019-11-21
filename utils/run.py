from tqdm import tqdm

from CollaborativeFilterDiego import CollaborativeFilter
from SLIM.SLIMRecommender import SLIMRecommender
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
        # Open up target_users file
        target_users_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users.csv"), "r")
        target_users = list(target_users_file)[1:]  # Eliminate header

        # Instantiate a new submission file
        recommendation_matrix_file_to_submit = open(
            os.path.join(ROOT_PROJECT_PATH, "data/recommendation_matrix_to_submit.csv"), "w")
        recommendation_matrix_file_to_submit.write("user_id,item_list\n")
        for user in tqdm(target_users):
            items_recommended = recommender.recommend(int(user))
            items_recommended = " ".join(str(x) for x in items_recommended)
            user = user.replace("\n", "")  # remove \n from user file
            recommendation_matrix_file_to_submit.write(user + "," + items_recommended + "\n")
        recommendation_matrix_file_to_submit.close()

    @staticmethod
    def run_test_recommender(recommender):
        evaluator = Evaluator()
        helper = Helper()
        MAP_final = 0.0
        # TODO URM_data contains the whole dataset and not the train set
        URM_train = helper.convert_URM_to_csr(helper.URM_data)

        # Fit URM with training fit data
        recommender.fit(URM_train)

        # Load target users
        target_users_test = helper.load_target_users_test()
        relevant_items = helper.load_relevant_items()

        for user in tqdm(target_users_test):
            recommended_items = recommender.recommend(int(user))
            relevant_item = relevant_items[int(user)]
            relevant_item = helper.convert_list_of_string_into_int(relevant_item)

            #print(recommended_items)

            MAP_final += evaluator.MAP(recommended_items, np.array(relevant_item))

        MAP_final /= len(target_users_test)
        return MAP_final

    @staticmethod
    def train_test_recommender(model, test_mode=True, parameters_set=True):

        helper = Helper()

        # TODO URM_data contains the whole dataset and not the train set
        URM_all = helper.convert_URM_to_csr(helper.URM_data)

        if test_mode:
            URM_train, URM_test, _, test_data = split_train_test(URM_all, 0.8)
        else:
            URM_train = URM_all

        if model == "top_popular":
            raise NotImplementedError

        elif model == "cbf":
            # TODO load ICM matrices
            raise NotImplementedError

        elif model == "collaborative_filter":
            recommender = CollaborativeFilter()
            recommender.fit(URM_train)
            RunRecommender.perform_evaluation(recommender, test_data, test_mode=test_mode)

        elif model == "SLIM":
            recommender = SLIMRecommender(URM_train)
            recommender.fit(epochs=1000)
            RunRecommender.perform_evaluation(recommender, test_data, test_mode=test_mode)

        else:
            print("No model called", model, "available")
            exit(-1)



    @staticmethod
    def perform_evaluation(recommender, test_data: dict, test_mode=False):

        if test_mode:
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

        else:
            RunRecommender.write_submission(recommender)


