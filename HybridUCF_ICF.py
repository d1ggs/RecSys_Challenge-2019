from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper
from TopPopularRecommender import TopPopRecommender

user_cf_parameters = {"topK": 410,
                      "shrink": 0}
item_cf_parameters = {"topK": 5,
                      "shrink": 8}


class HybridUCFICFRecommender():
    def __init__(self, weights):
        self.weights = weights
        self.user_cf = UserCollaborativeFilter(topK=user_cf_parameters["topK"], shrink=user_cf_parameters["shrink"])
        self.item_cf = ItemCollaborativeFilter(topK=item_cf_parameters["topK"], shrink=item_cf_parameters["shrink"])
        self.top_pop = TopPopRecommender()
        self.helper = Helper()
        self.cold_users = self.helper.get_cold_user_ids()


    def fit(self, URM_train):
        self.URM_train = URM_train
        # Fit the URM into single recommenders
        self.user_cf.fit(self.URM_train)
        self.item_cf.fit(self.URM_train)
        self.top_pop.fit(self.URM_train)
    def compute_scores(self, user_id):
        scores_user_cf = self.user_cf.compute_scores(user_id)
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores = (self.weights["user_cf"] * scores_user_cf) + (self.weights["item_cf"] * scores_item_cf)

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
        return recommended_items[0 : at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

if __name__ == "__main__":

    # Train and test data are now loaded by the helper

    weights_hybrid_ucf_icf = {"user_cf": 0.05, "item_cf": 0.95}

    hybrid_ucficf = HybridUCFICFRecommender(weights_hybrid_ucf_icf)

    # Evaluation is performed by RunRecommender
    RunRecommender.run(hybrid_ucficf)
