from utils.run import RunRecommender

if __name__ == '__main__':
    RunRecommender.train_test_recommender("collaborative_filter", test_mode=True)
