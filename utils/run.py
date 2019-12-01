from tqdm import tqdm
from utils.helper import Helper
import os
from evaluation.Evaluator import Evaluator

# Put root project dir in a constant
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RunRecommender:

    @staticmethod
    def run(recommender):

        # Helper contains methods to convert URM in CSR
        helper = Helper()
        URM_all = helper.URM_csr

        # Start recommendation
        recommender.fit(URM_all)

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
            items_recommended = recommender.recommend(int(user), exclude_seen=True)[:10]

            items_recommended = " ".join(str(x) for x in items_recommended)
            user = user.replace("\n", "")  # remove \n from user file
            recommendation_matrix_file_to_submit.write(user + "," + items_recommended + "\n")
        recommendation_matrix_file_to_submit.close()

    @staticmethod
    def evaluate_on_test_set(recommender_class, fit_parameters):

        MAP_final = 0.0
        evaluator, helper = Evaluator(), Helper()
        URM_train, eval_data = helper.URM_train_test, helper.test_data

        recommender = recommender_class(URM_train)
        recommender.fit(**fit_parameters)

        for user in tqdm(eval_data.keys()):
            recommended_items = recommender.recommend(int(user), exclude_seen=True)
            relevant_item = eval_data[int(user)]

            MAP_final += evaluator.MAP(recommended_items, relevant_item)

        MAP_final /= len(eval_data.keys())
        print("MAP-10 score:", MAP_final)
        MAP_final *= 0.665
        # TODO find new conversion factor for test set
        # print("MAP-10 public approx score:", MAP_final)

        return MAP_final

    @staticmethod
    def perform_evaluation(recommender):
        """Takes an already fitted recommender and evaluates on test data.
         If test_mode is false writes the submission"""

        print("Performing evaluation on test set...")

        MAP_final = 0.0
        evaluator, helper = Evaluator(),  Helper()
        URM_train, eval_data = helper.URM_train_eval, helper.eval_data

        recommender.fit(URM_train)
        for user in tqdm(eval_data.keys()):
            recommended_items = recommender.recommend(int(user), exclude_seen=True)
            relevant_item = eval_data[int(user)]

            MAP_final += evaluator.MAP(recommended_items, relevant_item)

        MAP_final /= len(eval_data.keys())
        print("MAP-10 score:", MAP_final)
        MAP_final *= 0.665
        print("MAP-10 public approx score:", MAP_final)

        return MAP_final

    @staticmethod
    def perform_evaluation_slim(recommender, test_data: dict):
        """Takes an already fitted recommender and evaluates on test data.
         If test_mode is false writes the submission"""

        if not test_data:
            print("Missing test data! Exiting...")
            exit(-1)

        print("Performing evaluation on test set...")

        MAP_final = 0.0
        evaluator = Evaluator()

        for user in tqdm(test_data.keys()):
            recommended_items = recommender.recommend(int(user), exclude_seen=True)[:10]
            relevant_item = test_data[int(user)]

            MAP_final += evaluator.MAP(recommended_items, relevant_item)

        MAP_final /= len(test_data.keys())
        MAP_final *= 0.665
        print("MAP-10 score:", MAP_final)
        return MAP_final
