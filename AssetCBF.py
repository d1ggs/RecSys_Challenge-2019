"""
Created on 09/11/2019

@author: Matteo Carretta
"""

import numpy as np
from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.helper import Helper
from utils.run import RunRecommender
from tqdm import tqdm
from copy import deepcopy


class AssetCBF:

    def __init__(self, URM):

        self.URM_train = URM

        self.helper = Helper()

    def compute_similarity_cbf(self, ICM, top_k, shrink, normalize=True, similarity="cosine"):
        # Compute similarities for weighted features recommender
        similarity_object = Compute_Similarity_Python(ICM.T, shrink=shrink, topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        w_sparse = similarity_object.compute_similarity()
        return w_sparse

    def fit(self, topK,shrink):

        self.topK = topK
        self.shrink = shrink

        # Load ICMs from helper
        self.ICM_asset = self.helper.bm25_normalization(self.helper.load_icm_asset())
        # Computing SMs
        self.SM_asset = self.compute_similarity_cbf(self.ICM_asset, top_k=self.topK, shrink=self.shrink)

    def compute_scores(self, user_id):
        users_list_train = self.URM_train[user_id]
        scores_asset = users_list_train.dot(self.SM_asset).toarray().ravel()

        return scores_asset

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

    parameters = { "topK": 4,
                   "shrink": 8}
    cbf_recommender = AssetCBF

    RunRecommender.evaluate_on_test_set(cbf_recommender, parameters)
    # RunRecommender.run(cbf_recommender)
