from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
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

    pset = [{"lr": 0.1, "epochs": 10, "top_k": 100},
            {"lr": 0.1, "epochs": 20, "top_k": 100},
            {"lr": 0.1, "epochs": 30, "top_k": 100},
            {"lr": 0.1, "epochs": 40, "top_k": 100},
            {"lr": 0.1, "epochs": 50, "top_k": 100},
            {"lr": 0.01, "epochs": 10, "top_k": 100},
            {"lr": 0.01, "epochs": 20, "top_k": 100},
            {"lr": 0.01, "epochs": 30, "top_k": 100},
            {"lr": 0.01, "epochs": 40, "top_k": 100},
            {"lr": 0.01, "epochs": 50, "top_k": 100},
            {"lr": 0.001, "epochs": 10, "top_k": 100},
            {"lr": 0.001, "epochs": 20, "top_k": 100},
            {"lr": 0.001, "epochs": 30, "top_k": 100},
            {"lr": 0.001, "epochs": 40, "top_k": 100},
            {"lr": 0.001, "epochs": 50, "top_k": 100},
            ]

    pset = [{"lr": 0.1, "epochs": 50, "top_k": 20},
            {"lr": 0.01, "epochs": 30, "top_k": 5}]

    helper = Helper()

    # Train and test data are now loaded by the helper
    URM_train, test_data = helper.get_train_test_data()

    map = []

    for params in pset:
        recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False, verbose=False)
        recommender.fit(epochs=params["epochs"], learning_rate=params["lr"], topK=params["top_k"])
        map.append(RunRecommender.perform_evaluation_slim(recommender, test_data))

    exit(1)

    plt.plot(map)
    plt.ylabel('MAP-10 score')
    plt.xlabel('Parameter set')
    plt.show()
