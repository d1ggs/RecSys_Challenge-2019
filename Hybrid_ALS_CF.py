from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
from TopPopularRecommender import TopPopRecommender
from ALS import AlternatingLeastSquare

import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper

user_cf_parameters = {"topK": 410,
                      "shrink": 0}
item_cf_parameters = {"topK": 29,
                      "shrink": 22}

als_parameters = {'n_factors': 680, 'regularization': 0.08905996811679368, 'iterations': 30}

class HybridALSCF(object):

    RECOMMENDER_NAME = "HybridElasticNetICF"

    def __init__(self, URM_train, mode="validation", ):
        self.URM_train = URM_train



        # Init single recommenders
        self.user_cf = UserCollaborativeFilter(URM_train)
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.top_pop = TopPopRecommender(URM_train)
        self.als = AlternatingLeastSquare(URM_train)
        # Get the cold users list
        self.cold_users, _ = Helper().compute_cold_warm_user_ids(URM_train)

        # Fit the single recommenders, this saves time in case of changing just the relative weights

        self.user_cf.fit(**user_cf_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.top_pop.fit()
        self.als.fit(**als_parameters)

    def fit(self, als_weight=0.0, user_cf_weight=0.2, item_cf_weight=0.3):

        # Normalize the weights, just in case
        weight_sum = item_cf_weight + user_cf_weight + als_weight

        self.weights = {"item_cf": item_cf_weight/weight_sum,
                        "user_cf": user_cf_weight/weight_sum,
                        "als": als_weight/weight_sum}

    def compute_scores(self, user_id):
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_user_cf = self.user_cf.compute_scores(user_id)
        scores_als = self.als.compute_scores(user_id)

        scores = (self.weights["item_cf"] * scores_item_cf) + \
                 (self.weights["user_cf"] * scores_user_cf) + \
                 (self.weights["als"] * scores_als)

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

    weights = {'als_weight': 0.0, 'item_cf_weight': 0.08340610330630326, 'user_cf_weight': 0.006456181309865703}
    hybrid_ucficf = HybridALSCF

    # Evaluation is performed by RunRecommender
    RunRecommender.evaluate_on_validation_set(hybrid_ucficf, weights)

    # RunRecommender.run(hybrid_ucficf, weights)