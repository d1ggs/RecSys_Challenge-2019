from utils.run import RunRecommender

if __name__ == '__main__':
    pset = [
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

    RunRecommender.train_test_recommender("collaborative_filter", parameters_set=pset, test_mode=True, split=False)
