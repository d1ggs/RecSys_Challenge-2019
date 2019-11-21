from utils.run import RunRecommender

if __name__ == '__main__':
    pset = [
        {"top_k": 50, "shrink": 0},
        {"top_k": 100, "shrink": 0},
        {"top_k": 150, "shrink": 0},
        {"top_k": 200, "shrink": 0},
        {"top_k": 250, "shrink": 0},
        {"top_k": 300, "shrink": 0},
        {"top_k": 350, "shrink": 0},
        {"top_k": 400, "shrink": 0},
        {"top_k": 450, "shrink": 0},
        {"top_k": 500, "shrink": 0},
        {"top_k": 50, "shrink": 2},
        {"top_k": 100, "shrink": 2},
        {"top_k": 150, "shrink": 2},
        {"top_k": 200, "shrink": 2},
        {"top_k": 250, "shrink": 2},
        {"top_k": 300, "shrink": 2},
        {"top_k": 350, "shrink": 2},
        {"top_k": 400, "shrink": 2},
        {"top_k": 450, "shrink": 2},
        {"top_k": 500, "shrink": 2},
    ]

    RunRecommender.train_test_recommender("collaborative_filter", parameters_set=pset, test_mode=True, split=True)
