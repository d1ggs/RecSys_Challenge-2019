from utils.run import RunRecommender

if __name__ == '__main__':
    pset = [
        {"top_k": 700, "shrink": 1},
        {"top_k": 800, "shrink": 1},
        {"top_k": 900, "shrink": 1},
        {"top_k": 700, "shrink": 2},
        {"top_k": 800, "shrink": 2},
        {"top_k": 900, "shrink": 2},
        {"top_k": 400, "shrink": 1},
        {"top_k": 500, "shrink": 1},
        {"top_k": 600, "shrink": 1},
        {"top_k": 400, "shrink": 2},
        {"top_k": 500, "shrink": 2},
        {"top_k": 600, "shrink": 2},
        {"top_k": 100, "shrink": 0},
        {"top_k": 200, "shrink": 0},
        {"top_k": 300, "shrink": 0},
        {"top_k": 400, "shrink": 0},
        {"top_k": 500, "shrink": 0},
        {"top_k": 600, "shrink": 0},
        {"top_k": 700, "shrink": 0},
        {"top_k": 800, "shrink": 0},
        {"top_k": 900, "shrink": 0}
    ]

    RunRecommender.train_test_recommender("collaborative_filter", parameters_set=[{"top_k": 800, "shrink": 0}], test_mode=False, split=False)
