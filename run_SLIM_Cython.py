from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from evaluation.Evaluator import Evaluator
from utils.helper import Helper

if __name__ == '__main__':

    pset5 = [{"lr": 0.001, "epochs": 30, "top_k": 5}]

    helper = Helper()

    # Train and test data are now loaded by the helper
    URM_train, test_data = helper.get_train_test_data(resplit=False, split_fraction=0, leave_out=1)

    evaluator = Evaluator()

    params = [{"lr": 0.1, "epochs": 1000, "top_k": 1},
              {"lr": 0.1, "epochs": 1000, "top_k": 2},
              {"lr": 0.1, "epochs": 1000, "top_k": 3},
              {"lr": 0.1, "epochs": 1000, "top_k": 4},
              {"lr": 0.1, "epochs": 1000, "top_k": 5},
              {"lr": 0.1, "epochs": 1000, "top_k": 6},
              {"lr": 0.1, "epochs": 1000, "top_k": 7},
              {"lr": 0.1, "epochs": 1000, "top_k": 8},
              {"lr": 0.1, "epochs": 1000, "top_k": 9}]

    map10 = []

    for p in params:
        early_stopping = {"validation_every_n": 10, "stop_on_validation": True,
                          "validation_metric": "MAP", "lower_validations_allowed": 2,
                          "evaluator_object": evaluator}
        recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False, verbose=False)

        best_MAP, best_epoch = recommender.fit(epochs=1000, learning_rate=p["lr"], topK=p["top_k"], random_seed=1234,
                                               **early_stopping)
        print("Parameters:", p)
        print("Best epoch:", best_epoch)
        print("MAP score:", best_MAP)
        map10.append((best_MAP, best_epoch))

    max_map = 0
    best_parameters = None
    best_epochs = 0

    for i in range(len(map10)):
        if map10[i][0] > max_map:
            max_map = map10[i][0]
            best_epochs = map10[i]
            best_parameters = params[i]

    print("Parameters:", best_parameters)
    print("Best epoch:", best_epochs)
    print("MAP score:", max_map)
    # RunRecommender.write_submission(recommender)
