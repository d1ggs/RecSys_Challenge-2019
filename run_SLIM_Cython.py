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

    pset1 = [{"lr": 0.001, "epochs": 30, "top_k": 1},
            {"lr": 0.001, "epochs": 30, "top_k": 2},
            {"lr": 0.001, "epochs": 30, "top_k": 3},
            {"lr": 0.001, "epochs": 30, "top_k": 4},
            {"lr": 0.001, "epochs": 30, "top_k": 5}]

    pset2 = [{"lr": 0.001, "epochs": 30, "top_k": 6},
            {"lr": 0.001, "epochs": 30, "top_k": 7},
            {"lr": 0.001, "epochs": 30, "top_k": 8},
            {"lr": 0.001, "epochs": 30, "top_k": 9},
            {"lr": 0.001, "epochs": 30, "top_k": 10}]

    pset3 = [{"lr": 0.001, "epochs": 30, "top_k": 20},
            {"lr": 0.001, "epochs": 30, "top_k": 30},
            {"lr": 0.001, "epochs": 30, "top_k": 40},
            {"lr": 0.001, "epochs": 30, "top_k": 50},
            {"lr": 0.001, "epochs": 30, "top_k": 60}]

    pset4 = [{"lr": 0.001, "epochs": 30, "top_k": 70},
            {"lr": 0.001, "epochs": 30, "top_k": 80},
            {"lr": 0.001, "epochs": 30, "top_k": 90},
            {"lr": 0.001, "epochs": 30, "top_k": 100},
            {"lr": 0.001, "epochs": 30, "top_k": 200}]

    pset5 = [{"lr": 0.001, "epochs": 30, "top_k": 5}]

    helper = Helper()

    # Train and test data are now loaded by the helper
    URM_train, test_data = helper.get_train_test_data_diego(resplit=True, split_fraction=0, leave_out=10)

    map = []

    for params in pset5:
        recommender = SLIM_BPR_Cython(URM_train, recompile_cython=False, verbose=False)
        recommender.fit(epochs=params["epochs"], learning_rate=params["lr"], topK=params["top_k"], random_seed=1234)
        #RunRecommender.write_submission(recommender)
        map.append(RunRecommender.perform_evaluation_slim(recommender, test_data))

    #plt.plot(map)
    #plt.ylabel('MAP-10 score')
    #plt.xlabel('Parameter set')
    #plt.show()
