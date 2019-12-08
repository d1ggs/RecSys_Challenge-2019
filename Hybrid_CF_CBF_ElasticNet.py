from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper
from TopPopularRecommender import TopPopRecommender
from UserBasedCBF import UserBasedCBF
from ItemBasedCBF import ItemBasedCBF
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
user_cf_parameters = {"topK": 410,
                      "shrink": 0}
item_cf_parameters = {"topK": 5,
                      "shrink": 8}
user_cbf_parameters = {"topK_age": 48,
                       "topK_region": 0,
                       "shrink_age": 8,
                       "shrink_region": 1,
                       "age_weight": 0.3}
item_cbf_parameters = {"topK_asset": 300,
                       "topK_price": 100,
                       "topK_sub_class": 80,
                       "shrink_asset": 4,
                       "shrink_price": 20,
                       "shrink_sub_class": 1,
                       "weight_asset": 0.33,
                       "weight_price": 0.33,
                       "weight_sub_class": 0.34}

slim_elastic_parameters = {"l1_ratio": 0.001, "positive_only": True, "topK": 1000}


class HybridUCFICFElasticNetRecommender():
    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.user_cf = UserCollaborativeFilter(URM_train)
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.top_pop = TopPopRecommender(URM_train)
        self.user_based_cbf = UserBasedCBF(URM_train)
        self.item_based_cbf = ItemBasedCBF(URM_train)
        self.slim_elastic = MultiThreadSLIM_ElasticNet(self.URM_train.copy())
        self.helper = Helper()
        self.cold_users = self.helper.get_cold_user_ids()

    def fit(self, user_cf=0.03, item_cf=0.71, user_cbf=0.05, item_cbf=0.06, slim_elastic=0.15):
        self.weights = {"user_cf": user_cf, "item_cf": item_cf, "user_cbf": user_cbf, "item_cbf": item_cbf, "slim_elastic": slim_elastic}

        # Fit the URM into single recommenders
        self.slim_elastic.fit(**slim_elastic_parameters)
        self.user_cf.fit(**user_cf_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.top_pop.fit()
        self.user_based_cbf.fit(**user_cbf_parameters)
        self.item_based_cbf.fit(**item_cbf_parameters)

    def compute_scores(self, user_id):
        scores_user_cf = self.user_cf.compute_scores(user_id)
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_user_cbf = self.user_based_cbf.compute_scores(user_id)
        scores_item_cbf = self.item_based_cbf.compute_scores(user_id)
        scores_slim_elastic = self.slim_elastic._compute_item_score(user_id_array=np.asarray(user_id)).squeeze()
        scores = (self.weights["user_cf"] * scores_user_cf) + (self.weights["item_cf"] * scores_item_cf) +\
                 (self.weights["user_cbf"] * scores_user_cbf) + (self.weights["item_cbf"] * scores_item_cbf) +\
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
        return recommended_items[0 : at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

if __name__ == "__main__":

    # Train and test data are now loaded by the helper
    hybrid = HybridUCFICFElasticNetRecommender

    weights_hybrid_ucf_icf = {"user_cf": 0.03, "item_cf": 0.81, "user_cbf": 0.04, "item_cbf": 0.02, "slim_elastic": 0.1}



    # Evaluation is performed by RunRecommender
    # RunRecommender.evaluate_on_test_set(hybrid, weights_hybrid_ucf_icf)

    RunRecommender.run(hybrid, weights_hybrid_ucf_icf)
