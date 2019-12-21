from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from TopPopularRecommender import TopPopRecommender
from RP3betaRecommender import RP3betaRecommender

import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper

user_cf_parameters = {'topK': 44, 'shrink': 5}
item_cf_parameters = {'topK': 3, 'shrink': 17}
rp3_parameters = {'alpha': 1.0043111988071665, 'beta': 0.0029837192341647823, 'implicit': False, 'min_rating': 0, 'normalize_similarity': False, 'topK': 680}


SLIM_parameters = {'alpha': 0.0023512567548654, 'l1_ratio': 0.0004093694334328875, 'positive_only': True, 'topK': 25}


class HybridElasticNetICFUCFRP3Beta(object):

    RECOMMENDER_NAME = "HybridElasticNetICF"

    def __init__(self, URM_train, mode="dataset"):
        self.URM_train = URM_train



        # Init single recommenders
        self.user_cf = UserCollaborativeFilter(URM_train)
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.top_pop = TopPopRecommender(URM_train)
        self.SLIM = MultiThreadSLIM_ElasticNet(URM_train)
        self.rp3 = RP3betaRecommender(URM_train)
        # Get the cold users list
        self.cold_users = Helper().get_cold_user_ids(mode)

        # Fit the single recommenders, this saves time in case of changing just the relative weights

        self.user_cf.fit(**user_cf_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.rp3.fit(**rp3_parameters)
        self.top_pop.fit()
        self.SLIM.fit(**SLIM_parameters)

    def fit(self, SLIM_weight=0.6719606935709935, user_cf_weight=0.08340610330630326, item_cf_weight=0.006456181309865703, rp3_weight = 0):

        # Normalize the weights, just in case
        weight_sum = item_cf_weight + SLIM_weight + user_cf_weight + rp3_weight

        self.weights = {"item_cf": item_cf_weight/weight_sum,
                        "SLIM": SLIM_weight/weight_sum,
                        "user_cf": user_cf_weight/weight_sum,
                        "rp3": rp3_weight/weight_sum}

    def compute_scores(self, user_id):
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_SLIM = self.SLIM._compute_item_score(user_id).squeeze()
        scores_user_cf = self.user_cf.compute_scores(user_id)
        scores_rp3 = self.rp3._compute_item_score(user_id).squeeze()
        scores = (self.weights["item_cf"] * scores_item_cf) + \
                 (self.weights["SLIM"] * scores_SLIM) + \
                 (self.weights["user_cf"] * scores_user_cf) +\
                 (self.weights["rp3"] * scores_rp3)

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

    weights = {'SLIM_weight': 0.8048223770755749, 'item_cf_weight': 0.21238894307574496, 'rp3_weight': 0.12587087079797205, 'user_cf_weight': 0.040353203361717556}


    hybrid_ucficf = HybridElasticNetICFUCFRP3Beta

    # Evaluation is performed by RunRecommender
    # RunRecommender.evaluate_on_test_set(hybrid_ucficf, weights)

    RunRecommender.run(hybrid_ucficf, weights)