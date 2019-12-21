import numpy as np

from RegionalTopPopRecommender import RegionalTopPopRecommender
from UserBasedCBF import UserBasedCBF


class HybridUserCBFRegionalTopPop(object):

    def __init__(self, URM_train, mode="dataset"):
        self.mode = mode
        self.URM_train = URM_train
        self.toppop = RegionalTopPopRecommender(URM_train)
        self.user_cbf = UserBasedCBF(URM_train)
        self.toppop_weight = 0.5

    def fit(self, top_pop_weight=0.5, topK=93 * 5, shrink=1, normalize=True, similarity="dice",
            suppress_interactions=True):
        self.user_cbf.fit(topK=topK, shrink=shrink, normalize=normalize, similarity=similarity,
                          suppress_interactions=suppress_interactions)
        self.toppop_weight = top_pop_weight

    def recommend(self, user_id, at=10, exclude_seen=True):
        scores = (1 - self.toppop_weight) * self.user_cbf.compute_scores(
            user_id) + self.toppop_weight * self.toppop.scores(user_id)

        # Filter to exclude already seen items
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
