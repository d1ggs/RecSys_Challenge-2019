from ItemBasedCBF import ItemBasedCBF
from UserCollaborativeFilter import UserCollaborativeFilter
import numpy as np

from TopPopularRecommender import TopPopRecommender
from utils.helper import Helper
from utils.run import RunRecommender

item_cbf_parameters = {"topK_asset": 100,
                       "topK_price": 100,
                       "topK_sub_class": 200,
                       "shrink_asset": 2,
                       "shrink_price": 2,
                       "shrink_sub_class": 0,
                       "weight_asset": 0.2,
                       "weight_price": 0.15,
                       "weight_sub_class": 0.65}
cb_parameters = {"topK": 210,
                 "shrink": 0}


class HybridCBCBFRecommender(object):

    RECOMMENDER_NAME = "HybridCBCBFRecommender"

    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.cbf = ItemBasedCBF(URM_train)
        self.cb = UserCollaborativeFilter(URM_train)
        self.toppop = TopPopRecommender(URM_train)
        self.cold_users = Helper().get_cold_user_ids()

    def fit(self, cbf_weight=0.4, cb_weight=0.6):
        weight_sum = cbf_weight + cb_weight
        self.cbf_weight = cbf_weight/weight_sum
        self.cb_weight = cb_weight/weight_sum
        # Fit the URM into single recommenders
        self.cbf.fit(**item_cbf_parameters)
        self.cb.fit(**cb_parameters)
        self.toppop.fit()

    def compute_scores(self, user_id):

        # Otherwise proceed with regular recommendations
        scores_cbf = self.cbf.compute_scores(user_id)
        scores_cb = self.cb.compute_scores(user_id)
        scores = (self.cbf_weight * scores_cbf) + (self.cb_weight * scores_cb)

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True):
        if user_id in self.cold_users and enable_toppop:
            # If the user has no interactions a TopPopular recommendations is still better than random
            recommended_items = self.toppop.recommend(user_id)
        else:
            scores = self.compute_scores(user_id)
            if exclude_seen:
                self.filter_seen(user_id, scores)
            recommended_items = np.argsort(scores)
            recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[0: at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":
    # Train and test data are now loaded by the helper

    weights_hybrid = {"cbf_weight": 0.4, "cb_weight": 0.6}

    hybrid_cbcbf = HybridCBCBFRecommender(weights_hybrid)

    # Evaluation is performed by RunRecommender
    RunRecommender.perform_evaluation(hybrid_cbcbf)
