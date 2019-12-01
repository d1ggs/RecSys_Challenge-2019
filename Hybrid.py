from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np

from evaluation.Evaluator import Evaluator
from utils.run import RunRecommender
from utils.helper import Helper
from TopPopularRecommender import TopPopRecommender
from UserBasedCBF import UserBasedCBF

user_cf_parameters = {"topK": 410,
                      "shrink": 0}
item_cf_parameters = {"topK": 5,
                      "shrink": 8}
user_cbf_parameters = {"topK_age": 990,
                       "topK_region": 978,
                       "shrink_age": 864,
                       "shrink_region": 821,
                       "age_weight":  0.9978304816546826}

user_cbf_parameters ={'topK_age': 981, 'topK_region': 983, 'shrink_region': 981, 'shrink_age': 74, 'age_weight': 0.1475020799647149, 'similarity': 'cosine', 'normalize': False}


class HybridUCFICFRecommender(object):
    RECOMMENDER_NAME = "HybridUCFICFRecommender"

    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.user_cf = UserCollaborativeFilter(topK=user_cf_parameters["topK"], shrink=user_cf_parameters["shrink"])
        self.item_cf = ItemCollaborativeFilter(topK=item_cf_parameters["topK"], shrink=item_cf_parameters["shrink"])
        self.top_pop = TopPopRecommender()
        self.userbasedcbf = UserBasedCBF(self.URM_train)
        self.helper = Helper()
        self.cold_users = self.helper.get_cold_user_ids()

    def fit(self, user_cf_weight=0.05, item_cf_weight=0.90, user_cbf_weight=0.05):
        # Fit the URM into single recommenders
        weight_sum = user_cf_weight + item_cf_weight + user_cbf_weight
        if weight_sum == 0:
            weight_sum = 1
        self.weights = {"user_cf": user_cf_weight/weight_sum, "item_cf": item_cf_weight/weight_sum, "user_cbf": user_cbf_weight/weight_sum}
        self.user_cf.fit(self.URM_train)
        self.item_cf.fit(self.URM_train)
        self.top_pop.fit(self.URM_train)
        self.userbasedcbf.fit(**user_cbf_parameters)

    def compute_scores(self, user_id):
        scores_user_cf = self.user_cf.compute_scores(user_id)
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_item_cbf = self.userbasedcbf.compute_scores(user_id)
        scores = (self.weights["user_cf"] * scores_user_cf) + (self.weights["item_cf"] * scores_item_cf) + \
                 (self.weights["user_cbf"] * scores_item_cbf)

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True):
        if user_id in self.cold_users and enable_toppop:
            # If the user has no interactions a TopPopular recommendation is still better than random
            recommended_items = self.top_pop.recommend(user_id)
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

    weights_hybrid_ucf_icf = {"user_cf_weight": 0, "item_cf_weight": 0.50, "user_cbf_weight": 0.50}

    hybrid_ucficf = HybridUCFICFRecommender

    # Evaluation is performed by RunRecommender
    RunRecommender.evaluate_on_test_set(hybrid_ucficf, weights_hybrid_ucf_icf)
    # RunRecommender.run(hybrid_ucficf)
