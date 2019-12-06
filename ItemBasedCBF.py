"""
Created on 09/11/2019

@author: Matteo Carretta
"""

import numpy as np
from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.helper import Helper
from utils.run import RunRecommender


class ItemBasedCBF:

    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.helper = Helper()

    def compute_similarity_cbf(self, ICM, top_k, shrink, normalize=True, similarity="cosine"):
        # Compute similarities for weighted features recommender
        similarity_object = Compute_Similarity_Python(ICM.T, shrink=shrink, topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        w_sparse = similarity_object.compute_similarity()
        return w_sparse

    def fit(self, topK_asset=100, topK_price=100, topK_sub_class=200, shrink_asset=2, shrink_price=2,
                 shrink_sub_class=0, weight_asset=0.2, weight_price=0.15, weight_sub_class=0.65):

        self.topK_asset = topK_asset
        self.topK_price = topK_price
        self.topK_sub_class = topK_sub_class
        self.shrink_asset = shrink_asset
        self.shrink_price = shrink_price
        self.shrink_sub_class = shrink_sub_class
        self.weight_asset = weight_asset
        self.weight_price = weight_price
        self.weight_sub_class = weight_sub_class

        # Load ICMs from helper
        self.ICM_asset = self.helper.load_icm_asset()
        self.ICM_asset = self.helper.tfidf_normalization(self.ICM_asset)

        self.ICM_price = self.helper.load_icm_price()
        self.ICM_sub_class = self.helper.load_icm_sub_class()
        # Computing SMs
        self.SM_asset = self.compute_similarity_cbf(self.ICM_asset, top_k=self.topK_asset, shrink=self.shrink_asset)
        self.SM_prince = self.compute_similarity_cbf(self.ICM_price, top_k=self.topK_price, shrink=self.shrink_price)
        self.SM_sub_class = self.compute_similarity_cbf(self.ICM_sub_class, top_k=self.topK_sub_class, shrink=self.shrink_sub_class)

    def compute_scores(self, user_id):
        users_list_train = self.URM_train[user_id]
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
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

if __name__ == "__main__":

    # evaluator.split_data_randomly()

    parameters = {"topK_asset": 100,
                   "topK_price": 300,
                   "topK_sub_class": 80,
                   "shrink_asset": 20,
                   "shrink_price": 20,
                   "shrink_sub_class": 0,
                   "weight_asset": 0.35,
                   "weight_price": 0,
                   "weight_sub_class": 0.65}
    cbf_recommender = ItemBasedCBF

    RunRecommender.evaluate_on_test_set(cbf_recommender, parameters)
    # RunRecommender.run(cbf_recommender)
