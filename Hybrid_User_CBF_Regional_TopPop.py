import numpy as np

from RegionalTopPopRecommender import RegionalTopPopRecommender
from UserCBF import UserCBF
from utils.helper import Helper
from utils.run import RunRecommender


class HybridUserCBFRegionalTopPop(object):

    RECOMMENDER_NAME = "ColdUsersHybrid"

    def __init__(self, URM_train, mode="dataset"):
        self.mode = mode
        self.URM_train = URM_train

    def fit(self, top_pop_weight=0.02139131367609725, topK=765, shrink=6, normalize=True, bm_25_norm=False, similarity="jaccard",
            suppress_interactions=False, asymmetric_alpha=0.5):
        if bm_25_norm:
            self.URM_train = Helper().bm25_normalization(self.URM_train)

        self.toppop = RegionalTopPopRecommender(self.URM_train)
        self.user_cbf = UserCBF(self.URM_train)

        self.user_cbf.fit(topK=topK, shrink=shrink, normalize=normalize, similarity=similarity,
                          suppress_interactions=suppress_interactions, asymmetric_alpha=asymmetric_alpha)
        self.toppop.fit()
        self.toppop_weight = top_pop_weight

    def recommend(self, user_id, at=10, exclude_seen=True):
        scores = (1 - self.toppop_weight) * self.user_cbf.compute_scores(
            user_id) + self.toppop_weight * self.toppop.scores(user_id)

        # Filter to exclude already seen items
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[0:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == '__main__':

    new_params_cosine = {'normalize': False, 'shrink': 50, 'suppress_interactions': False, 'topK': 540, 'top_pop_weight': 0.0020283643469209793}
    params_test_bm_25 = {'bm_25_norm': True}
    RunRecommender.evaluate_on_test_set(HybridUserCBFRegionalTopPop, {}, Kfold=10, user_group="cold", parallelize_evaluation=True, parallel_fit=False)
    RunRecommender.evaluate_on_test_set(HybridUserCBFRegionalTopPop, new_params_cosine, Kfold=10, user_group="cold", parallelize_evaluation=True, parallel_fit=False)
