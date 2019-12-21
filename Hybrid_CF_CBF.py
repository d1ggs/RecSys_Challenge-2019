from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper
from TopPopularRecommender import TopPopRecommender
from UserBasedCBF import UserBasedCBF
from ItemBasedCBF import ItemBasedCBF

user_cf_parameters = {"topK": 410,
                      "shrink": 0}
item_cf_parameters = {"topK": 29,
                      "shrink": 22}
user_cbf_parameters = {"topK_age": 48,
                       "topK_region": 0,
                       "shrink_age": 8,
                       "shrink_region": 1,
                       "age_weight": 0.3}
item_cbf_parameters = {"topK_asset": 300,
                       "topK_price": 100,
                       "topK_sub_class": 80,
                       "shrink_asset": 4,
                       "shrink_price": 20,
                       "shrink_sub_class": 1,
                       "weight_asset": 0.33,
                       "weight_price": 0.33,
                       "weight_sub_class": 0.34}


class HybridUCFICFRecommender(object):

    RECOMMENDER_NAME = "HybridUCFICFRecommender"

    def __init__(self, URM_train):
        self.URM_train = URM_train

        # Init single recommenders
        self.user_cf = UserCollaborativeFilter(URM_train)
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.top_pop = TopPopRecommender(URM_train)
        self.user_based_cbf = UserBasedCBF(URM_train)
        self.item_based_cbf = ItemBasedCBF(URM_train)

        # Get the cold users list
        self.cold_users = Helper().get_cold_user_ids("dataset")

        # Fit the single recommenders
        self.user_cf.fit(**user_cf_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.top_pop.fit()
        self.user_based_cbf.fit(**user_cbf_parameters)
        self.item_based_cbf.fit(**item_cbf_parameters)

    def fit(self, user_cf_weight=0.03, item_cf_weight=0.88, user_cbf_weight=0.07, item_cbf_weight=0.02):

        # Normalize the weights, just in case
        weight_sum = user_cbf_weight + user_cf_weight + item_cbf_weight + item_cf_weight

        self.weights = {"user_cf": user_cf_weight/weight_sum,
                        "item_cf": item_cf_weight/weight_sum,
                        "user_cbf": user_cbf_weight/weight_sum,
                        "item_cbf": item_cbf_weight/weight_sum}

    def compute_scores(self, user_id):
        scores_user_cf = self.user_cf.compute_scores(user_id)
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_user_cbf = self.user_based_cbf.compute_scores(user_id)
        scores_item_cbf = self.item_based_cbf.compute_scores(user_id)
        scores = (self.weights["user_cf"] * scores_user_cf) + (self.weights["item_cf"] * scores_item_cf) + \
                 (self.weights["user_cbf"] * scores_user_cbf) + (self.weights["item_cbf"] * scores_item_cbf)

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True):
        if user_id in self.cold_users and enable_toppop:
            # If the user has no interactions a TopPopular recommendation is still better than random
            recommended_items = self.top_pop.recommend(user_id)
        else:
            # Otherwise compute regular recommendations
            scores = self.compute_scores(user_id)
            if exclude_seen:
                self.filter_seen(user_id, scores)
            recommended_items = np.argsort(scores)
            recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[0: at]

    def filter_seen(self, user_id, scores):
        """Remove items that are in the user profile from recommendations

        :param user_id: array of user ids for which to compute recommendations
        :param scores: array containing the scores for each object"""

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":
    # Train and test data are now loaded by the helper

    weights_hybrid_ucf_icf = {"user_cf_weight": 0.03, "item_cf_weight": 0.88, "user_cbf_weight": 0.07, "item_cbf_weight": 0.02}

    hybrid_ucficf = HybridUCFICFRecommender

    # Evaluation is performed by RunRecommender
    RunRecommender.evaluate_on_test_set(hybrid_ucficf, weights_hybrid_ucf_icf)
    # RunRecommender.run(hybrid_ucficf)
