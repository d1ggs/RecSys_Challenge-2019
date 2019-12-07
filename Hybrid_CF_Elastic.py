from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper
from TopPopularRecommender import TopPopRecommender
from UserBasedCBF import UserBasedCBF
from ItemBasedCBF import ItemBasedCBF
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

item_cf_parameters = {"topK": 5,
                      "shrink": 8}

slim_elastic_parameters = {"l1_ratio": 2.2767573538452768e-05, "positive_only": True, "topK": 100}


class HybridICFElastic():
    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.top_pop = TopPopRecommender(URM_train)
        self.item_based_cbf = ItemBasedCBF(URM_train)
        self.slim_elastic = MultiThreadSLIM_ElasticNet(self.URM_train.copy())
        self.helper = Helper()
        self.cold_users = self.helper.get_cold_user_ids("test")

    def fit(self, slim_elastic):
        self.weight_item_cf = {"item_cf": 1-slim_elastic, "slim_elastic": slim_elastic}

        # Fit the URM into single recommenders
        self.slim_elastic.fit(**slim_elastic_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.top_pop.fit()

    def compute_scores(self, user_id):
        scores_item_cf = self.item_based_cf.compute_scores(user_id)
        scores_slim_elastic = self.slim_elastic._compute_item_score(user_id_array=np.asarray(user_id)).squeeze()
        scores = (self.weights["item_cf"] * scores_item_cf) + (self.weights["slim_elastic"] * scores_slim_elastic.squeeze())

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
    hybrid = HybridICFElastic

    weights_hybrid_icf_elastic = {"slim_elastic": 0.5}



    # Evaluation is performed by RunRecommender
    # RunRecommender.evaluate_on_test_set(hybrid, weights_hybrid_ucf_icf)

    RunRecommender.run(hybrid, weights_hybrid_icf_elastic)
