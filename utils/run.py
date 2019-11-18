from tqdm import tqdm
from utils.helper import Helper
import os
from evaluation.Evaluator import Evaluator
import numpy as np

# Put root project dir in a constant
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RunRecommender:

    @staticmethod
    def run(recommender):
        # Open up target_users file
        target_users_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users.csv"), "r")

        # Helper contains methods to convert URM in CSR
        helper = Helper()
        URM_CSR = helper.convert_URM_to_csr(helper.URM_data)

        # Start recommendation
        recommender.fit(URM_CSR)

        # Instantiate a new submission file
        target_users = list(target_users_file)[1:] # Eliminate header
        recommendation_matrix_file_to_submit = open(os.path.join(ROOT_PROJECT_PATH, "data/recommendation_matrix_to_submit.csv"), "w")
        recommendation_matrix_file_to_submit.write("user_id,item_list\n")
        for user in tqdm(target_users):
            items_recommended = recommender.recommend(int(user))
            items_recommended = " ".join(str(x) for x in items_recommended)
            user = user.replace("\n", "") # remove \n from user file
            recommendation_matrix_file_to_submit.write(user + "," + items_recommended + "\n")
        recommendation_matrix_file_to_submit.close()


    @staticmethod
    def run_test_recommender(recommender, refit=True):
        evaluator = Evaluator()
        helper = Helper()
        MAP_final = 0.0
        URM_train = helper.convert_URM_to_csr(helper.URM_data)
        # Fit URM with training fit data
        recommender.fit(URM_train)
        URM_test = helper.load_URM_test_csr()
        # Load target users
        target_users_test = helper.load_target_users_test()
        relevant_items = helper.load_relevant_items()
        for user in tqdm(target_users_test):
            recommended_items = recommender.recommend(int(user), URM_test)
            relevant_item = relevant_items[int(user)]
            relevant_item = helper.convert_list_of_string_into_int(relevant_item)

            MAP_final += evaluator.MAP(np.asarray(recommended_items), np.asarray(relevant_item))


        MAP_final /= len(target_users_test)
        return MAP_final

