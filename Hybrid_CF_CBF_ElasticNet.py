from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper
from TopPopularRecommender import TopPopRecommender
from ItemBasedCBF import ItemCBF
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

user_cf_parameters = {"topK": 410,
                      "shrink": 0}
item_cf_parameters = {"topK": 29,
                      "shrink": 22}
# item_cbf_parameters = {"topK": 1,
#                        "shrink": 12}
item_cbf_parameters = {"topK": 200,
                       "shrink": 5}

SLIM_parameters = {'alpha': 0.0023512567548654, 'l1_ratio': 0.0004093694334328875, 'positive_only': True, 'topK': 25,
                   'random_state': 1234}


class HybridCFCBFElasticNet():
    def __init__(self, URM_train, mode="dataset"):
        self.URM_train = URM_train
        self.user_cf = UserCollaborativeFilter(URM_train)
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.top_pop = TopPopRecommender(URM_train)
        self.item_based_cbf = ItemCBF(URM_train)
        self.slim_elastic = MultiThreadSLIM_ElasticNet(self.URM_train.copy())
        self.helper = Helper()
        self.cold_users = self.helper.get_cold_user_ids(mode)

        # Fit the URM into single recommenders, save time in case weights should be changed
        self.slim_elastic.fit(**SLIM_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.top_pop.fit()
        self.item_based_cbf.fit(**item_cbf_parameters)

    def fit(self, item_cf_weight=0.71, item_cbf_weight=0.06, slim_elastic_weight=0.15):

        weight_sum = item_cbf_weight + item_cf_weight + slim_elastic_weight

        self.weights = {"item_cf": item_cf_weight / weight_sum,
                        "item_cbf": item_cbf_weight / weight_sum,
                        "slim_elastic": slim_elastic_weight / weight_sum}

    def compute_scores(self, user_id):
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_item_cbf = self.item_based_cbf.compute_scores(user_id)
        scores_slim_elastic = self.slim_elastic._compute_item_score(user_id_array=np.asarray(user_id)).squeeze()
        scores = (self.weights["item_cf"] * scores_item_cf) + (self.weights["item_cbf"] * scores_item_cbf) +\
                 (self.weights["slim_elastic"] * scores_slim_elastic.squeeze())

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
        return recommended_items[0:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":
    # Train and test data are now loaded by the helper
    hybrid = HybridCFCBFElasticNet

    weights_hybrid_ucf_icf = {'item_cbf_weight': 0.033099412622983986, 'item_cf_weight': 0.31062121622263944, 'slim_elastic_weight': 0.7264945269296246}



    # Evaluation is performed by RunRecommender
    # RunRecommender.evaluate_on_test_set(hybrid, weights_hybrid_ucf_icf)

    RunRecommender.run(hybrid, weights_hybrid_ucf_icf)