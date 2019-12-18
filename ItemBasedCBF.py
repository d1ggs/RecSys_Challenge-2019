"""
Created on 09/11/2019

@author: Matteo Carretta
"""

import numpy as np
from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.helper import Helper
from utils.run import RunRecommender
from AssetCBF import AssetCBF
from PriceCBF import PriceCBF
from SubClassCBF import SubClassCBF

asset_cbf_parameters = {"topK": 200,
                        "shrink": 5} # to fit with optimal values

price_cbf_parameters = {"topK": 200,
                        "shrink": 5}

sub_class_cbf_parameters = {"topK": 200,
                            "shrink": 5}

class ItemBasedCBF:

    def __init__(self, URM_train, mode="dataset"):
        self.URM_train = URM_train
        self.helper = Helper()
        self.asset_cbf = AssetCBF(self.URM_train)
        self.price_cbf = PriceCBF(self.URM_train)
        self.sub_class_cbf = SubClassCBF(self.URM_train)

    def compute_similarity_cbf(self, ICM, top_k, shrink, normalize=True, similarity="cosine"):
        # Compute similarities for weighted features recommender
        similarity_object = Compute_Similarity_Python(ICM.T, shrink=shrink, topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        w_sparse = similarity_object.compute_similarity()
        return w_sparse

    def fit(self, weight_asset = 0.41617098566013966,
                  weight_price = 0.36833811086509627,
                  weight_sub_class = 0.6977028008004889):

        self.asset_cbf.fit(**asset_cbf_parameters)
        self.price_cbf.fit(**price_cbf_parameters)
        self.sub_class_cbf.fit(**sub_class_cbf_parameters)
        sum = weight_asset + weight_price + weight_sub_class
        self.weight_asset = weight_asset/sum
        self.weight_price = weight_price/sum
        self.weight_sub_class = weight_sub_class/sum


    def compute_scores(self, user_id):
        scores_asset = self.asset_cbf.compute_scores(user_id)
        scores_price = self.price_cbf.compute_scores(user_id)
        scores_sub_class = self.sub_class_cbf.compute_scores(user_id)

        scores = (scores_asset * self.weight_asset) + (scores_price * self.weight_price) + (scores_sub_class * self.weight_sub_class)
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

    parameters = {"weight_asset": 0.3333,
                  "weight_price": 0.3333,
                  "weight_sub_class": 0.3333}
    cbf_recommender = ItemBasedCBF

    RunRecommender.evaluate_on_test_set(cbf_recommender, parameters)
    # RunRecommender.run(cbf_recommender)
