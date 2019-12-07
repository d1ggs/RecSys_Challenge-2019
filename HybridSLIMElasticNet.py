from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np

from TopPopularRecommender import TopPopRecommender
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

from utils.helper import Helper
from utils.run import RunRecommender

SLIM_parameters = {'alpha': 0.003890771067122292, 'l1_ratio': 2.2767573538452768e-05, 'positive_only': True}

class HybridSLIMElasticNetTopPop(object):
    RECOMMENDER_NAME = "HybridUCFICFRecommender"

    def __init__(self, URM_train, mode="dataset"):
        self.URM_train = URM_train
        self.top_pop = TopPopRecommender(URM_train)
        self.SLIMElasticNet = MultiThreadSLIM_ElasticNet(URM_train)
        self.cold_users = Helper().get_cold_user_ids(mode)

    def fit(self):
        self.SLIMElasticNet.fit(**SLIM_parameters)
        self.top_pop.fit()

    def compute_scores(self, user_id):
        scores_SLIM = self.SLIMElasticNet._compute_item_score(user_id).squeeze()
        scores = scores_SLIM

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
    recommender = HybridSLIMElasticNetTopPop
    #RunRecommender.evaluate_on_eval_set(HybridSLIMElasticNetTopPop, {})
    # Evaluation is performed by RunRecommender
    RunRecommender.run(recommender, {})

