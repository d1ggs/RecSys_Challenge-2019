from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from evaluation.Evaluator import Evaluator
from utils.helper import Helper
from utils.run import RunRecommender

import matplotlib.pyplot as plt

if __name__ == '__main__':
    pset_collaborative_filter = [
        {"top_k": 500, "shrink": 0},
        {"top_k": 550, "shrink": 0},
        {"top_k": 600, "shrink": 0},
        {"top_k": 650, "shrink": 0},
        {"top_k": 700, "shrink": 0},
        {"top_k": 750, "shrink": 0},
        {"top_k": 800, "shrink": 0},
        {"top_k": 850, "shrink": 0},
        {"top_k": 900, "shrink": 0},
        {"top_k": 950, "shrink": 0},
        {"top_k": 500, "shrink": 2},
        {"top_k": 550, "shrink": 2},
        {"top_k": 600, "shrink": 2},
        {"top_k": 650, "shrink": 2},
        {"top_k": 700, "shrink": 2},
        {"top_k": 750, "shrink": 2},
        {"top_k": 800, "shrink": 2},
        {"top_k": 850, "shrink": 2},
        {"top_k": 900, "shrink": 2},
        {"top_k": 900, "shrink": 2},
    ]

    pset1 = [{"lr": 0.1, "epochs": 10, "top_k": 5},
             {"lr": 0.1, "epochs": 20, "top_k": 5},
             {"lr": 0.1, "epochs": 30, "top_k": 5},
             {"lr": 0.1, "epochs": 40, "top_k": 5},
             {"lr": 0.1, "epochs": 50, "top_k": 5},
             {"lr": 0.1, "epochs": 60, "top_k": 5},
             {"lr": 0.1, "epochs": 70, "top_k": 5},
             {"lr": 0.1, "epochs": 80, "top_k": 5},
             {"lr": 0.1, "epochs": 90, "top_k": 5},
             {"lr": 0.1, "epochs": 100, "top_k": 5}]

    pset2 = [{"lr": 0.005, "epochs": 10, "top_k": 5},
             {"lr": 0.005, "epochs": 20, "top_k": 5},
             {"lr": 0.005, "epochs": 30, "top_k": 5},
             {"lr": 0.005, "epochs": 40, "top_k": 5},
             {"lr": 0.005, "epochs": 50, "top_k": 5},
             {"lr": 0.005, "epochs": 60, "top_k": 5},
             {"lr": 0.005, "epochs": 70, "top_k": 5},
             {"lr": 0.005, "epochs": 80, "top_k": 5},
             {"lr": 0.005, "epochs": 90, "top_k": 5},
             {"lr": 0.005, "epochs": 100, "top_k": 5}]

    pset3 = [{"lr": 0.001, "epochs": 50, "top_k": 5},
             {"lr": 0.001, "epochs": 60, "top_k": 5},
             {"lr": 0.001, "epochs": 70, "top_k": 5},
             {"lr": 0.001, "epochs": 80, "top_k": 5},
             {"lr": 0.001, "epochs": 90, "top_k": 5},
             {"lr": 0.001, "epochs": 100, "top_k": 5},
             {"lr": 0.001, "epochs": 120, "top_k": 5},
             {"lr": 0.001, "epochs": 140, "top_k": 5},
             {"lr": 0.001, "epochs": 160, "top_k": 5},
             {"lr": 0.001, "epochs": 180, "top_k": 5},
             {"lr": 0.001, "epochs": 200, "top_k": 5},
             ]


    pset5 = [{"lr": 0.001, "epochs": 30, "top_k": 5}]

    helper = Helper()

    # Train and test data are now loaded by the helper
    URM_train, test_data = helper.get_train_test_data(resplit=False, split_fraction=0, leave_out=1)

    evaluator = Evaluator()

    pset2 = {"lr": 0.001, "epochs": 1000, "top_k": 5}
    pset = {"lr": 0.001, "epochs": 1000, "top_k": 100}

    recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False, verbose=False)

    recommender.fit(epochs=1000, learning_rate=pset["lr"], topK=pset["top_k"], random_seed=1234,
                    **{"validation_every_n": 10, "stop_on_validation": True,
                       "validation_metric" : "MAP", "lower_validations_allowed" : 2,
                       "evaluator_object" : evaluator})
    # RunRecommender.write_submission(recommender)
