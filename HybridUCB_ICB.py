from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np
from utils.run import RunRecommender


user_cb_parameters = {"topK": 510,
                      "shrink": 4}
item_cb_parameters = {"topK": 2,
                      "shrink": 20}


class HybridUCBICBRecommender():
    def __init__(self, weights):
        self.weights = weights
        self.user_cb = UserCollaborativeFilter(topK=user_cb_parameters["topK"], shrink=user_cb_parameters["shrink"])
        self.item_cb = ItemCollaborativeFilter(topK=item_cb_parameters["topK"], shrink=item_cb_parameters["shrink"])



    def fit(self, URM_train):
        self.URM_train = URM_train
        # Fit the URM into single recommenders
        self.user_cb.fit(self.URM_train)
        self.item_cb.fit(self.URM_train)

    def compute_scores(self, user_id):
        scores_user_cb = self.user_cb.compute_scores(user_id)
        scores_item_cb = self.item_cb.compute_scores(user_id)
        scores = (self.weights["user_cb"] * scores_user_cb) + (self.weights["item_cb"] * scores_item_cb)

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

    weights_hybrid_ucb_icb = {"user_cb": 0.0, "item_cb": 1}

    hybrid_ucbicb = HybridUCBICBRecommender(weights_hybrid_ucb_icb)

    # Evaluation is performed by RunRecommender
    RunRecommender.perform_evaluation(hybrid_ucbicb)
