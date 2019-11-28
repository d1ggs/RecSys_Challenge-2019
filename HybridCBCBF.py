from ItemBasedCBF import ItemBasedCBF
from UserCollaborativeFilter import CollaborativeFilter
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


class HybridCBCBFRecommender():
    def __init__(self, weights):
        self.weights = weights
        self.cbf = ItemBasedCBF(topK_asset=item_cbf_parameters["topK_asset"], topK_price=item_cbf_parameters["topK_price"],
                                topK_sub_class=item_cbf_parameters["topK_sub_class"], shrink_asset=item_cbf_parameters["shrink_asset"],
                                shrink_price=item_cbf_parameters["shrink_price"], shrink_sub_class=item_cbf_parameters["shrink_sub_class"],
                                weight_asset=item_cbf_parameters["weight_asset"], weight_price=item_cbf_parameters["weight_price"],
                                weight_sub_class=item_cbf_parameters["weight_sub_class"])
        self.cb = CollaborativeFilter(topK=cb_parameters["topK"], shrink=cb_parameters["shrink"])
        self.toppop = TopPopRecommender()
        self.cold_users = Helper().get_cold_user_ids()



    def fit(self, URM_train):
        self.URM_train = URM_train
        # Fit the URM into single recommenders
        self.cbf.fit(self.URM_train)
        self.cb.fit(self.URM_train)
        self.toppop.fit(self.URM_train)

    def compute_scores(self, user_id):
        if user_id in self.cold_users:
            # If the user has no interactions a TopPopular recommendations is still better than random
            scores = self.toppop.recommend(user_id)
        else:
            # Otherwise proceed with regular recommendations
            scores_cbf = self.cbf.compute_scores(user_id)
            scores_cb = self.cb.compute_scores(user_id)
            scores = (self.weights["cbf"] * scores_cbf) + (self.weights["cb"] * scores_cb)

        return scores


    def recommend(self, user_id, at=10, exclude_seen=True):
        scores = self.compute_scores(user_id)
        if exclude_seen:
            self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[0 : at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":

    # Train and test data are now loaded by the helper

    weights_hybrid = {"cbf": 0.4, "cb": 0.6}

    hybrid_cbcbf = HybridCBCBFRecommender(weights_hybrid)

    # Evaluation is performed by RunRecommender
    RunRecommender.perform_evaluation(hybrid_cbcbf)
