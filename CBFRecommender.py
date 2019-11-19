"""
Created on 09/11/2019

@author: Matteo Carretta
"""

import numpy as np
from base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.helper import Helper
from utils.run import RunRecommender
from evaluation.Evaluator import Evaluator
from tqdm import tqdm
from copy import deepcopy


class CBFRecomender:

    def __init__(self, knn_asset=100, knn_price=100, knn_sub_class=100, shrink_asset=2, shrink_price=2,
                 shrink_sub_class=2, weight_asset=0.4, weight_price=0.5, weight_sub_class=0.1):

        self.knn_asset = knn_asset
        self.knn_price = knn_price
        self.knn_sub_class = knn_sub_class
        self.shrink_asset = shrink_asset
        self.shrink_price = shrink_price
        self.shrink_sub_class = shrink_sub_class
        self.weight_asset = weight_asset
        self.weight_price = weight_price
        self.weight_sub_class = weight_sub_class
        self.helper = Helper()

    def compute_similarity_cbf(self, ICM, top_k, shrink, normalize=True, similarity="cosine"):
        # Compute similarities for weighted features recommender
        similarity_object = Compute_Similarity_Python(ICM.T, shrink=shrink, topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        w_sparse = similarity_object.compute_similarity()
        return w_sparse

    def fit(self, URM):

        # URM Loading
        self.URM = URM

        # Load ICMs from helper
        self.ICM_asset = self.helper.load_icm_asset()
        self.ICM_price = self.helper.load_icm_price()
        self.ICM_sub_class = self.helper.load_icm_sub_class()

        # Computing SMs
        self.SM_asset = self.compute_similarity_cbf(self.ICM_asset, top_k=self.knn_asset, shrink=self.shrink_asset)
        self.SM_prince = self.compute_similarity_cbf(self.ICM_price, top_k=self.knn_price, shrink=self.shrink_price)
        self.SM_sub_class = self.compute_similarity_cbf(self.ICM_sub_class, top_k=self.knn_sub_class,
                                                        shrink=self.shrink_sub_class)

    def compute_scores(self, user_id):
        users_list_train = self.URM[user_id]
        scores_asset = users_list_train.dot(self.SM_asset).toarray().ravel()
        scores_price = users_list_train.dot(self.SM_prince).toarray().ravel()
        scores_sub_class = users_list_train.dot(self.SM_sub_class).toarray().ravel()

        scores = (scores_asset * self.weight_asset) + (scores_price * self.weight_price) + (
                scores_sub_class * self.weight_sub_class)
        return scores

    def recommend(self, user_id, at=10, exclude_seen=True):
        # Compute scores of the recommendation
        scores = self.compute_scores(user_id)

        # Filter to exclude already seen items
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def find_best_fit(self, parameter_list):
        print("Looking for best parameter set over ", len(parameter_list))
        results = []
        best = 0
        best_parameters = None
        best_model = None

        # Collect MAP scores for each parameter set

        for parameter_set in tqdm(parameter_list):
            self.__init__(knn_asset=parameter_set["knn_asset"], knn_price=parameter_set["knn_price"],
                          knn_sub_class=parameter_set["knn_sub_class"], shrink_asset=parameter_set["shrink_asset"],
                          shrink_price=parameter_set["shrink_price"],
                          shrink_sub_class=parameter_set["shrink_sub_class"],
                          weight_asset=parameter_set["weight_asset"], weight_price=parameter_set["weight_price"],
                          weight_sub_class=parameter_set["weight_sub_class"])

            result = RunRecommender.run_test_recommender(cbf_recommender)
            results.append(result)

            if result > best:
                best_parameters = parameter_set
                best_model = deepcopy(self)
                best = result

        print("Best MAP score obtained: ", best)
        return results, best_model, best_parameters


if __name__ == "__main__":
    cbf_recommender = CBFRecomender()
    evaluator = Evaluator()
    evaluator.split_data_randomly()

    parameters = [{"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.5, "weight_sub_class": 0.1},

                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.3, "weight_sub_class": 0.3},

                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.2, "weight_sub_class": 0.4}]

    #cbf_recommender.find_best_fit(parameters)

    map_10 = RunRecommender.run_test_recommender(cbf_recommender)
    print(map_10)
    RunRecommender.run(cbf_recommender)
