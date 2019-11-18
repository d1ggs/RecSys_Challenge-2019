import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper
from evaluation.Evaluator import Evaluator


class TopPopRecommender:

  def __init__(self, helper):
    self.helper = helper

  def fit(self, URM_CSR):
    """ --- CSR Matrix Computation ---
        Training data loading and its conversion to a CSR matrix is done in run.py with the functions provided by Helper
    """
    self.URM_data = self.helper.URM_data
    self.URM_CSR = URM_CSR
    """ --- Actual Popularity computation --- """
    # Calculate item popularity by summing for each item the rating of every use
    # The most popular playlist is the one with more songs in it
    item_popularity = (self.URM_CSR > 0).sum(axis=0)
    # Squeeze term removes single-dimensional entries from the shape of an array.
    item_popularity = np.array(item_popularity).squeeze()

    # Ordering the items according to the popularity values
    self.popular_items = np.argsort(item_popularity)

    # Flip order high to lower
    self.popular_items = np.flip(self.popular_items, axis=0)

  def recommend(self, user_id, at=10, remove_seen = True):
    if remove_seen:
      unseen_items_mask = np.in1d(self.popular_items, self.URM_CSR[user_id].indices,  assume_unique=True, invert=True)

      unseen_items = self.popular_items[unseen_items_mask]

      recommended_items = unseen_items[0:at]

    else:
      recommended_items = self.popularItems[0:at]

    return recommended_items



if __name__ == "__main__":
  helper = Helper()
  top_popular = TopPopRecommender(helper)

  RunRecommender.run(top_popular)
  #print(map10)
