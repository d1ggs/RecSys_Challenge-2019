from UserCollaborativeFilter import UserCollaborativeFilter
from ItemCollaborativeFilter import ItemCollaborativeFilter
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from TopPopularRecommender import TopPopRecommender

import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper

user_cf_parameters = {"topK": 410,
                      "shrink": 0}
item_cf_parameters = {"topK": 29,
                      "shrink": 22}
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
# SLIM_parameters = {'alpha': 0.003890771067122292, 'l1_ratio': 2.2767573538452768e-05, 'positive_only': True, 'topK': 100}
SLIM_parameters = {'alpha': 0.0023512567548654,
                   'l1_ratio': 0.0004093694334328875,
                   'positive_only': True, 'topK': 25,
                   'random_state': 1234
                   }


class HybridElasticNetICFUCF(object):
    RECOMMENDER_NAME = "HybridElasticNetICF"

    def __init__(self, URM_train, mode="dataset"):
        self.URM_train = URM_train

        self.user_data_dict, self.cluster_dict, self.region_dict, self.age_dict = \
            Helper().get_demographic_info(verbose=False, mode=mode)

        self.cluster_recommendations = {}
        self.region_recommendations = {}
        self.age_recommendations = {}

        # Init single recommenders
        self.user_cf = UserCollaborativeFilter(URM_train)
        self.item_cf = ItemCollaborativeFilter(URM_train)
        self.top_pop = TopPopRecommender(URM_train)
        self.SLIM = MultiThreadSLIM_ElasticNet(URM_train)

        # Get the cold users list
        self.cold_users = Helper().get_cold_user_ids(mode)

        # Fit the single recommenders, this saves time in case of changing just the relative weights

        self.user_cf.fit(**user_cf_parameters)
        self.item_cf.fit(**item_cf_parameters)
        self.top_pop.fit()
        self.SLIM.fit(**SLIM_parameters)

    def fit(self, SLIM_weight=0.6719606935709935, item_cf_weight=0.08340610330630326, user_cf_weight=0.006456181309865703):

        # Normalize the weights, just in case
        weight_sum = item_cf_weight + SLIM_weight + user_cf_weight

        self.weights = {"item_cf": item_cf_weight / weight_sum,
                        "SLIM": SLIM_weight / weight_sum,
                        "user_cf": user_cf_weight / weight_sum}

    def compute_scores(self, user_id):
        scores_item_cf = self.item_cf.compute_scores(user_id)
        scores_SLIM = self.SLIM._compute_item_score(user_id).squeeze()
        scores_user_cf = self.user_cf.compute_scores(user_id)

        scores = (self.weights["item_cf"] * scores_item_cf) + \
                 (self.weights["SLIM"] * scores_SLIM) + \
                 self.weights["user_cf"] * scores_user_cf

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True, use_demographic=True):
        # TODO check if elected users are cold users or if something is wrong
        if user_id in self.user_data_dict and use_demographic:
            print("got it")
            if user_id not in self.cold_users:
                print("diocane")
            obj = self.user_data_dict[user_id]
            if obj.cluster is not None and obj.age is None and obj.region is None:
                cluster = obj.cluster
                if cluster in self.cluster_recommendations:
                    scores = self.cluster_recommendations[cluster]
                else:
                    user_list = self.cluster_dict[cluster]
                    scores = np.zeros(shape=self.URM_train.shape[1])
                    for user in user_list:
                        scores += self.SLIM._compute_item_score(user).squeeze()
                        # scores += self.user_cf.compute_scores(user)
                        # scores += self.item_cf.compute_scores(user)
                    self.cluster_recommendations[cluster] = scores

            elif obj.cluster is None and obj.age is not None and obj.region is None:
                age = obj.age
                if age in self.age_recommendations:
                    scores = self.age_recommendations[age]
                else:
                    user_list = self.age_dict[age]
                    scores = np.zeros(shape=self.URM_train.shape[1])
                    for user in user_list:
                        scores += self.SLIM._compute_item_score(user).squeeze()
                        # scores += self.user_cf.compute_scores(user)
                        # scores += self.item_cf.compute_scores(user)
                    self.age_recommendations[age] = scores

            elif obj.cluster is None and obj.age is None and obj.region is not None:
                region = obj.region
                if region in self.region_recommendations:
                    scores = self.region_recommendations[region]
                else:
                    user_list = self.region_dict[region]
                    scores = np.zeros(shape=self.URM_train.shape[1])
                    for user in user_list:
                        scores += self.SLIM._compute_item_score(user).squeeze()
                        # scores += self.user_cf.compute_scores(user)
                        # scores += self.item_cf.compute_scores(user)
                    self.region_recommendations[region] = scores
            else:
                print("Something went wrong")
                exit()

            if exclude_seen:
                self.filter_seen(user_id, scores)
            recommended_items = np.argsort(scores)
            recommended_items = np.flip(recommended_items, axis=0)

        elif user_id in self.cold_users and enable_toppop:
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

    weights = {'SLIM_weight': 0.6719606935709935, 'item_cf_weight': 0.08340610330630326,
               'user_cf_weight': 0.006456181309865703}
    hybrid_ucficf = HybridElasticNetICFUCF

    # Evaluation is performed by RunRecommender
    # RunRecommender.evaluate_on_validation_set(hybrid_ucficf, weights)

    RunRecommender.run(hybrid_ucficf, weights)
