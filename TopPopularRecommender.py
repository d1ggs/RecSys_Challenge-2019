import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper


class TopPopRecommender:

    def __init__(self, URM_CSR, mode=""):
        self.URM_CSR = URM_CSR
        self.item_popularity = None

    def fit(self):
        # --- CSR Matrix Computation ---
        #Training data loading and its conversion to a CSR matrix is done in run.py with the functions provided by Helper

        # --- Actual Popularity computation ---
        # Calculate item popularity by summing for each item the rating of every use
        # The most popular playlist is the one with more songs in it

        item_popularity = (self.URM_CSR > 0).sum(axis=0)
        # Squeeze term removes single-dimensional entries from the shape of an array.
        self.item_popularity = np.array(item_popularity).squeeze()

        # Ordering the items according to the popularity values
        self.popular_items = np.argsort(self.item_popularity)

        # Flip order high to lower
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, at=10, exclude_seen=True):
        if exclude_seen:
            unseen_items_mask = np.in1d(self.popular_items, self.URM_CSR[user_id].indices, assume_unique=True, invert=True)

            unseen_items = self.popular_items[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popular_items[0:at]

        return recommended_items


if __name__ == "__main__":

    # evaluator = Evaluator()
    # evaluator.split_data_randomly_2()

    top_popular = TopPopRecommender

    map10 = RunRecommender.evaluate_on_test_set(top_popular, {}, Kfold=4, parallel_fit=True, sequential_MAP=False, user_group="cold")

    #print('{0:.128f}'.format(map10))
    print(map10)
