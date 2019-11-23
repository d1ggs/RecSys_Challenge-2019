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
    def train_test_recommender(model, test_mode=True, parameters_set=None, split=False):

        helper = Helper()

        # TODO URM_data contains the whole dataset and not the train set
        URM_all = helper.convert_URM_to_csr(helper.URM_data)

        # Check if we need to split the data to perform evaluation of the model
        if test_mode:
            URM_train, URM_test, test_data = split_train_test(URM_all, 0.8, rewrite=split)
        else:
            URM_train = URM_all
            test_data = None

        # Prepare for multiple evaluations
        if parameters_set:
            i = 1
            best_map = 0
            best_parameters = None
        else:
            raise NotImplementedError

        if model == "top_popular":
            raise NotImplementedError

        elif model == "cbf":
            # TODO load ICM matrices
            raise NotImplementedError

        elif model == "collaborative_filter":

            raise NotImplementedError

            # Fit and validate for each parameter set

            # for pset in parameters_set:
            #     print("---------------------------------------------------------------------------------")
            #     if test_mode:
            #         print("Computing performance of parameter set", i)
            #     # TODO Validate that the dictionary contains the correct keys
            #     recommender = CollaborativeFilter(URM_train)
            #     recommender.fit(topK=pset["top_k"], shrink=pset["shrink"])
            #     map10 = RunRecommender.perform_evaluation(recommender, test_data, test_mode=test_mode)
            #
            #     if test_mode:
            #         if map10 > best_map:
            #             best_map = map10
            #             best_parameters = pset
            #     i += 1
            #
            # if test_mode:
            #     print("Best MAP score:", best_map)
            #     print("Best parameters:")
            #     for k in best_parameters.keys():
            #         print(k, ":", best_parameters[k])


        elif model == "SLIM":

            # Fit and validate for each parameter set

            for pset in parameters_set:
                print("---------------------------------------------------------------------------------")
                if test_mode:
                    print("Computing performance of parameter set", i)
                # TODO Validate that the dictionary contains the correct keys
                recommender = SLIMRecommender(URM_train)
                recommender.fit(epochs=pset["epochs"], learning_rate=pset["lr"], top_k=pset["top_k"])
                map10 = RunRecommender.perform_evaluation(recommender, test_data, test_mode=test_mode)

                if test_mode:
                    if map10 > best_map:
                        best_map = map10
                        best_parameters = pset
                i += 1

            if test_mode:
                print("Best MAP score:", best_map)
                print("Best parameters:")
                for k in best_parameters.keys():
                    print(k, ":", best_parameters[k])

        elif model == "SLIM_cython":

            for pset in parameters_set:
                print("---------------------------------------------------------------------------------")
                if test_mode:
                    print("Computing performance of parameter set", i)
                # TODO Validate that the dictionary contains the correct keys
                recommender = SLIM_BPR_Cython(URM_train, recompile_cython=True)
                recommender.fit(epochs=pset["epochs"], learning_rate=pset["lr"], topK= pset["top_k"])
                map10 = RunRecommender.perform_evaluation(recommender, test_data, test_mode=test_mode)

                if test_mode:
                    if map10 > best_map:
                        best_map = map10
                        best_parameters = pset
                i += 1

            recommender = SLIM_BPR_Cython(URM_train, recompile_cython=True)
            recommender.fit()

            if test_mode:
                print("Best MAP score:", best_map)
                print("Best parameters:")
                for k in best_parameters.keys():
                    print(k, ":", best_parameters[k])


        else:
            print("No model called", model, "available")
            exit(-1)

    @staticmethod
    def perform_evaluation(recommender, test_data: dict, test_mode=False):
        """Takes an already fitted recommender and evaluates on test data.
         If test_mode is false writes the submission"""

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
            return MAP_final

        else:
            RunRecommender.write_submission(recommender)
            return True

