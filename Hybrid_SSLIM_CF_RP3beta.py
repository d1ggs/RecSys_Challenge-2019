from ItemBasedCBF import ItemCBF
from ItemCollaborativeFilter import ItemCollaborativeFilter
from SSLIMElasticNetRecommender import MultiThreadSSLIM_ElasticNet, SSLIMElasticNetRecommender
from Hybrid_User_CBF_Regional_TopPop import HybridUserCBFRegionalTopPop
from RP3betaRecommender import RP3betaRecommender

import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper

# item_cf_parameters = {"topK": 29,
#                       "shrink": 22}

# New best parameters
item_cf_parameters = {'shrink': 46.0, 'similarity': "jaccard", 'topK': 8}

rp3_parameters = {'alpha': 0.31932803725825626,
                  'beta': 0.19051435359666555,
                  'implicit': True, 'min_rating': 0,
                  'normalize_similarity': True,
                  'topK': 56}

item_cbf_parameters = {'normalize': True,
                       'shrink': 5,
                       'similarity': 'cosine',
                       'topK': 200}

SSLIM_parameters = {'alpha': 0.0024081648139725204,
                   'l1_ratio': 0.0007553368138338653,
                   'positive_only': False,
                   'side_alpha': 3.86358712510434,
                   'topK': 65}


class HybridSSLIMICFUCFRP3Beta(object):

    RECOMMENDER_NAME = "HybridElasticNetCFRP3Beta"

    def __init__(self, URM_train, mode="dataset", multithreaded=False, id=None):
        self.URM_train = URM_train

        # Init single recommenders
        self.item_cbf = ItemCBF(URM_train)
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.hybrid_cold = HybridUserCBFRegionalTopPop(URM_train)
        if multithreaded:
            self.SSLIM = MultiThreadSSLIM_ElasticNet(URM_train)
        else:
            self.SSLIM = SSLIMElasticNetRecommender(URM_train)
        self.rp3 = RP3betaRecommender(URM_train)
        # Get the cold users list
        self.cold_users, _ = Helper().compute_cold_warm_user_ids(URM_train)

        # Fit the single recommenders, this saves time in case of changing just the relative weights

        self.item_cbf.fit(**item_cbf_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.rp3.fit(**rp3_parameters)
        self.hybrid_cold.fit()
        self.SSLIM.fit(**SSLIM_parameters)

    def fit(self, SSLIM_weight=0.8950096358670148, item_cbf_weight=0.034234727663263104, item_cf_weight=0.011497379340447589, rp3_weight = 0.8894480634395567):

        # Normalize the weights, just in case
        weight_sum = item_cf_weight + SSLIM_weight + item_cbf_weight + rp3_weight

        self.weights = {"item_cf": item_cf_weight/weight_sum,
                        "SSLIM": SSLIM_weight / weight_sum,
                        "item_cbf": item_cbf_weight/weight_sum,
                        "rp3": rp3_weight/weight_sum}

    def compute_scores(self, user_id):
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_SSLIM = self.SSLIM._compute_item_score(user_id).squeeze()
        scores_item_cbf = self.item_cbf.compute_scores(user_id)
        scores_rp3 = self.rp3._compute_item_score(user_id).squeeze()
        scores = (self.weights["item_cf"] * scores_item_cf) + \
                 (self.weights["SSLIM"] * scores_SSLIM) + \
                 (self.weights["item_cbf"] * scores_item_cbf) +\
                 (self.weights["rp3"] * scores_rp3)

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True):
        if user_id in self.cold_users and enable_toppop:
            # If the user has no interactions a TopPopular recommendation is still better than random
            recommended_items = self.hybrid_cold.recommend(user_id)
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

    weights = {}

    hybrid_ucficf = HybridSSLIMICFUCFRP3Beta

    RunRecommender.evaluate_on_test_set(hybrid_ucficf, weights, Kfold=4, parallelize_evaluation=True)

    RunRecommender.run(hybrid_ucficf, weights)
