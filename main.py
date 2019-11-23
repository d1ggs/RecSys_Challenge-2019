from utils.run import RunRecommender

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

    # pset = [{"lr": 0.1, "epochs": 10, "top_k": 100},
    #         {"lr": 0.1, "epochs": 20, "top_k": 100},
    #         {"lr": 0.1, "epochs": 30, "top_k": 100},
    #         {"lr": 0.1, "epochs": 40, "top_k": 100},
    #         {"lr": 0.1, "epochs": 50, "top_k": 100},
    #         {"lr": 0.01, "epochs": 10, "top_k": 100},
    #         {"lr": 0.01, "epochs": 20, "top_k": 100},
    #         {"lr": 0.01, "epochs": 30, "top_k": 100},
    #         {"lr": 0.01, "epochs": 40, "top_k": 100},
    #         {"lr": 0.001, "epochs": 50, "top_k": 100},
    #         {"lr": 0.001, "epochs": 10, "top_k": 100},
    #         {"lr": 0.001, "epochs": 20, "top_k": 100},
    #         {"lr": 0.001, "epochs": 30, "top_k": 100},
    #         {"lr": 0.001, "epochs": 40, "top_k": 100},
    #         {"lr": 0.001, "epochs": 50, "top_k": 100},
    #         ]

    pset = [
    {"lr": 0.1, "epochs": 50, "top_k": 20}]


    RunRecommender.train_test_recommender("SLIM_cython", parameters_set=pset, test_mode=True, split=False)
