import multiprocessing

from tqdm import tqdm
from utils.helper import Helper
import os
from evaluation.Evaluator import Evaluator

# Put root project dir in a constant
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_mean_and_variance(data: list):
    total_sum = 0

    for val in data:
        total_sum += val

    mean = total_sum / len(data)

    for val in data:
        total_sum += (val - mean) ** 2

    variance = total_sum / (len(data) - 1)

    return mean, variance


def prepare_cold_users(URM_train, evaluation_data):
    cold_users, _ = Helper().compute_cold_warm_user_ids(URM_train)
    cold_users = list(cold_users)

    ok_users = []
    for user in cold_users:
        if user in evaluation_data.keys():
            ok_users.append(user)

    cold_users = set(ok_users)

    return cold_users

def prepare_warm_users(URM_train, evaluation_data):
    _, warm_users = Helper().compute_cold_warm_user_ids(URM_train)
    warm_users = list(warm_users)

    ok_users = []
    for user in warm_users:
        if user in evaluation_data.keys():
            ok_users.append(user)

    warm_users = set(ok_users)

    return warm_users


def fit_recommender(params):
    recommender_class = params[0]
    URM = params[1]
    id = params[2]
    fit_parameters = params[3]
    mode = params[4]
    recommender = recommender_class(URM)
    recommender.fit(**fit_parameters)

    return recommender, id


class RunRecommender(object):

    @staticmethod
    def run(recommender_class, fit_parameters, init_params=None, at=10, submission_path="data/recommendation_matrix_to_submit.csv"):

        # Helper contains methods to convert URM in CSR

        if init_params is None:
            init_params = {}

        URM_all = Helper().URM_csr

        recommender = recommender_class(URM_all, **init_params)
        recommender.fit(**fit_parameters)

        RunRecommender.write_submission(recommender, submission_path=submission_path, at=at)

    @staticmethod
    def write_submission(recommender, submission_path: str, at: int):
        submission_target = os.path.join(ROOT_PROJECT_PATH, submission_path)
        print("Writing submission in", submission_target)

        # Open up target_users file
        target_users_file = open(os.path.join(ROOT_PROJECT_PATH, "data/target_users.csv"), "r")
        target_users = list(target_users_file)[1:]  # Eliminate header

        # Instantiate a new submission file
        recommendation_matrix_file_to_submit = open(submission_target, 'w')
        recommendation_matrix_file_to_submit.write("user_id,item_list\n")
        for user in tqdm(target_users):
            items_recommended = recommender.recommend(int(user), exclude_seen=True, at=at)

            items_recommended = " ".join(str(x) for x in items_recommended)
            user = user.replace("\n", "")  # remove \n from user file
            recommendation_matrix_file_to_submit.write(user + "," + items_recommended + "\n")
        recommendation_matrix_file_to_submit.close()

    @staticmethod
    def evaluate_profile(recommender, users_to_evaluate: set):

        evaluator = Evaluator()

        MAP_final, _ = evaluator.evaluateRecommender(recommender, users_to_evaluate)

        print("MAP-10 score:", MAP_final)

        return MAP_final

    @staticmethod
    def evaluate_on_test_set(recommender_class, fit_parameters, init_params=None, users_to_evaluate=None, Kfold=0,
                             parallelize_evaluation=False,
                             parallel_fit=False, user_group="all"):

        if init_params is None:
            init_params = {}

        evaluator = Evaluator(test_mode=True)

        if Kfold > 0:
            MAP_final = 0
            fitted_recommenders = []

            num_cores = multiprocessing.cpu_count()

            kfold_data = Helper().get_kfold_data(Kfold)
            URMs = [data[1] for data in kfold_data]
            test_data_list = [data[3] for data in kfold_data]
            users_to_evaluate_list = []
            for data in kfold_data:
                if user_group == "cold":
                    users_to_evaluate_list.append(prepare_cold_users(data[1], data[3]))
                elif user_group == "warm":
                    users_to_evaluate_list.append(prepare_warm_users(data[1], data[3]))
                else:
                    users_to_evaluate_list.append(list(data[3].keys()))

            if parallel_fit and Kfold <= num_cores:

                if init_params is not None:
                    raise NotImplementedError("Can't handle init params in parallel fit")

                print("Parallelize fitting recommenders...")

                with multiprocessing.Pool(processes=num_cores) as p:
                    fitted_recommenders = p.map(fit_recommender,
                                                [(recommender_class, URM, recommender_id, fit_parameters, "test") for
                                                 URM, recommender_id in
                                                 zip(URMs, range(len(URMs)))])
                p.close()

                print("Done!")
            else:

                for i in range(Kfold):
                    URM_test = URMs[i]

                    recommender = recommender_class(URM_test, **init_params)
                    recommender.fit(**fit_parameters)

                    fitted_recommenders.append((recommender, i))

            data_list = []

            if parallelize_evaluation and Kfold <= num_cores:
                with multiprocessing.Pool(processes=Kfold) as p:
                    results = p.starmap(Evaluator().evaluate_recommender_kfold,
                                        [(recommender, users_to_evaluate_list[recommender_id],
                                          test_data_list[recommender_id])
                                         for recommender, recommender_id in fitted_recommenders])


                    for i in range(len(results)):
                        data_list.append(results[i][0])

                p.close()
            else:
                for recommender, recommender_id in fitted_recommenders:
                    MAP, _ = evaluator.evaluate_recommender_kfold(recommender,
                                                                  users_to_evaluate_list[recommender_id],
                                                                  test_data_list[recommender_id])
                    data_list.append(MAP)

            MAP_final, variance = compute_mean_and_variance(data_list)
            print("Variance over k-fold:", variance)

        else:
            URM_train = Helper().URM_train_test

            recommender = recommender_class(URM_train, **init_params)
            recommender.fit(**fit_parameters)

            if user_group == "cold":
                MAP_final, _ = evaluator.evaluate_recommender_on_cold_users(recommender)
            else:
                MAP_final, _ = evaluator.evaluateRecommender(recommender, users_to_evaluate)

        print("MAP-10 score:", MAP_final)

        return MAP_final

    @staticmethod
    def evaluate_on_validation_set(recommender_class, fit_parameters, init_params=None, user_group="all", users_to_evaluate=None, Kfold=0,
                                   parallel_fit=False, parallelize_evaluation=False):

        if init_params is None:
            init_params = {}

        evaluator = Evaluator(test_mode=False)

        if Kfold > 0:
            MAP_final = 0
            fitted_recommenders = []

            num_cores = multiprocessing.cpu_count()

            kfold_data = Helper().get_kfold_data(Kfold)
            URMs = [data[0] for data in kfold_data]
            validation_data_list = [data[2] for data in kfold_data]
            users_to_evaluate_list = []
            for data in kfold_data:
                if user_group == "cold":
                    users_to_evaluate_list.append(prepare_cold_users(data[0], data[2]))
                elif user_group == "warm":
                    users_to_evaluate_list.append(prepare_warm_users(data[0], data[2]))
                else:
                    users_to_evaluate_list.append(list(data[2].keys()))

            if parallel_fit and Kfold <= num_cores:

                if init_params is not None:
                    raise NotImplementedError("Can't handle init params in parallel fit")

                print("Parallelize fitting recommenders...")

                with multiprocessing.Pool(processes=Kfold) as p:
                    fitted_recommenders = p.map(fit_recommender,
                                                [(recommender_class, URM, recommender_id, fit_parameters, "validation")
                                                 for
                                                 URM, recommender_id in
                                                 zip(URMs, range(len(URMs)))])
                p.close()

                print("Done!")

            else:
                for i in range(Kfold):
                    URM_validation, _, validation_data, _ = Helper().get_kfold_data(Kfold)[i]

                    recommender = recommender_class(URM_validation, **init_params)
                    recommender.fit(**fit_parameters)

                    fitted_recommenders.append((recommender, i))

            data_list = []

            if parallelize_evaluation and Kfold <= num_cores:

                with multiprocessing.Pool(processes=Kfold) as p:
                    results = p.starmap(Evaluator().evaluate_recommender_kfold,
                                        [(recommender, users_to_evaluate_list[recommender_id],
                                          validation_data_list[recommender_id])
                                         for recommender, recommender_id in fitted_recommenders])

                    for i in range(len(results)):
                        data_list.append(results[i][0])

                p.close()
            else:
                for recommender, recommender_id in fitted_recommenders:
                    MAP, _ = evaluator.evaluate_recommender_kfold(recommender,
                                                                  users_to_evaluate_list[recommender_id],
                                                                  validation_data_list[recommender_id])
                    data_list.append(MAP)

            MAP_final, variance = compute_mean_and_variance(data_list)
            print("Variance over k-fold:", variance)


        else:
            URM_train = Helper().URM_train_validation

            recommender = recommender_class(URM_train, **init_params)
            recommender.fit(**fit_parameters)

            if user_group == "cold":
                MAP_final, _ = evaluator.evaluate_recommender_on_cold_users(recommender)
            else:
                MAP_final, _ = evaluator.evaluateRecommender(recommender, users_to_evaluate)

        print("MAP-10 score:", MAP_final)

        return MAP_final

    @staticmethod
    def perform_evaluation(recommender):
        """Takes an already fitted recommender and evaluates on test data.
         If test_mode is false writes the submission"""

        print("Performing evaluation on test set...")

        MAP_final = 0.0
        evaluator, helper = Evaluator(), Helper()
        URM_train, eval_data = helper.URM_train_validation, helper.validation_data

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

    @staticmethod
    def evaluate_hybrid_weights_test(recommender, weights, exclude_users=None):

        recommender.fit(**weights)

        MAP_final, _ = Evaluator(test_mode=True).evaluateRecommender(recommender, exclude_users)

        print("MAP-10 score:", MAP_final)

        return MAP_final

    @staticmethod
    def evaluate_hybrid_weights_validation(recommender, weights, exclude_users=None):

        recommender.fit(**weights)

        MAP_final, _ = Evaluator().evaluateRecommender(recommender, exclude_users)

        print("MAP-10 score:", MAP_final)

        return MAP_final

    @staticmethod
    def evaluate_hybrid_weights_test_kfold(recommender_list, weights, kfold=4, parallel_fit=False, user_group="all",
                                           parallelize_evaluation=False):

        num_cores = multiprocessing.cpu_count()

        users_to_evaluate_list = []
        test_data_list = []
        for i in range(kfold):
            _, URM_test, _, test_data = Helper().get_kfold_data(kfold)[i]
            test_data_list.append(test_data)

            if user_group == "cold":
                users_to_evaluate_list.append(prepare_cold_users(URM_test, test_data))
            elif user_group == "warm":
                raise NotImplementedError
            else:
                users_to_evaluate_list.append(list(test_data.keys()))

            recommender_list[i].fit(**weights)

        data_list = []

        if parallelize_evaluation and kfold <= num_cores:

            with multiprocessing.Pool(processes=kfold) as p:
                results = p.starmap(Evaluator().evaluate_recommender_kfold,
                                    [(recommender_list[i], users_to_evaluate_list[i], test_data_list[i])
                                     for i in range(kfold)])
                for i in range(len(results)):
                    data_list.append(results[i][0])
            p.close()
        else:
            for i in range(kfold):
                recommender = recommender_list[i]
                # recommender.fit(**weights)

                MAP, _ = Evaluator().evaluate_recommender_kfold(recommender, users_to_evaluate_list[i],
                                                                test_data_list[i])

                data_list.append(MAP)

        MAP_final, variance = compute_mean_and_variance(data_list)
        print("Variance over k-fold:", variance)
        print("MAP-10 score:", MAP_final)

        return MAP_final

    @staticmethod
    def evaluate_hybrid_weights_validation_kfold(recommender_list, weights, kfold=4, parallel_fit=False,
                                                 user_group="all", parallelize_evaluation=False):

        num_cores = multiprocessing.cpu_count()

        users_to_evaluate_list = []
        validation_data_list = []
        for i in range(kfold):
            URM_validation, _, validation_data, _ = Helper().get_kfold_data(kfold)[i]
            validation_data_list.append(validation_data)

            if user_group == "cold":
                users_to_evaluate_list.append(prepare_cold_users(URM_validation, validation_data))
            elif user_group == "warm":
                raise NotImplementedError
            else:
                users_to_evaluate_list.append(list(validation_data.keys()))

            recommender_list[i].fit(**weights)

        data_list = []

        if parallelize_evaluation and kfold <= num_cores:

            with multiprocessing.Pool(processes=kfold) as p:
                results = p.starmap(Evaluator().evaluate_recommender_kfold,
                                    [(recommender_list[i], users_to_evaluate_list[i], validation_data_list[i])
                                     for i in range(kfold)])
                for i in range(len(results)):
                    data_list.append(results[i][0])
            p.close()
        else:
            for i in range(kfold):
                recommender = recommender_list[i]
                # recommender.fit(**weights)

                MAP, _ = Evaluator().evaluate_recommender_kfold(recommender, users_to_evaluate_list[i],
                                                                validation_data_list[i])

                data_list.append(MAP)

        MAP_final, variance = compute_mean_and_variance(data_list)
        print("Variance over k-fold:", variance)
        print("MAP-10 score:", MAP_final)

        return MAP_final
