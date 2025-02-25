"""
Created on 09/11/2019

@author: Matteo Carretta
"""

import numpy as np
from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
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
        print(users_list_train)
        scores_asset = users_list_train.dot(self.SM_asset).toarray().ravel()
        scores_price = users_list_train.dot(self.SM_prince).toarray().ravel()
        scores_sub_class = users_list_train.dot(self.SM_sub_class).toarray().ravel()

        scores = (scores_asset * self.weight_asset) + (scores_price * self.weight_price) + (
                scores_sub_class * self.weight_sub_class)
        return scores

    def recommend(self, user_id, at=10, remove_seen=False):
        # Compute scores of the recommendation
        scores = self.compute_scores(user_id)

        # Filter to exclude already seen items
        if remove_seen:
            scores = self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        recommended_items = recommended_items[:10]
        return recommended_items

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":
    cbf_recommender = CBFRecomender()
    evaluator = Evaluator()
    # evaluator.split_data_randomly()
    k = [0, 25, 50, 75, 100]
    s = [0, 2, 4, 6, 8]

    parameters = [{"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.3, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.2, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.1, "weight_sub_class": 0.5},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0, "weight_sub_class": 0.6},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.5, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.5, "weight_price": 0.2, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.6, "weight_price": 0.1, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0, "weight_sub_class": 0.6},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0, "weight_price": 0.5, "weight_sub_class": 0.5},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.1, "weight_sub_class": 0.8},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.2, "weight_sub_class": 0.7},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.7, "weight_sub_class": 0},

                  {"knn_asset": k[0], "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": k[1], "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": k[2], "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": k[3], "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": k[4], "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},

                  {"knn_asset": 100, "knn_price": k[0], "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": k[1], "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": k[2], "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": k[3], "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": k[4], "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},

                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": k[0], "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": k[1], "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": k[2], "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": k[3], "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": k[4], "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},

                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": s[0], "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": s[1], "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": s[2], "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": s[3], "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": s[4], "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},

                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": s[0],
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": s[1],
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": s[2],
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": s[3],
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": s[4],
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},

                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": s[0], "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": s[1], "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": s[2], "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": s[3], "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": s[4], "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},


                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.5, "weight_sub_class": 0.1},
                  {"knn_asset": 90, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.3, "weight_sub_class": 0.3},
                  {"knn_asset": 80, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.2, "weight_sub_class": 0.4},
                  {"knn_asset": 70, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.1, "weight_sub_class": 0.5},
                  {"knn_asset": 60, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0, "weight_sub_class": 0.6},
                  {"knn_asset": 50, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.5, "weight_sub_class": 0.3},
                  {"knn_asset": 40, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 30, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.5, "weight_price": 0.2, "weight_sub_class": 0.3},
                  {"knn_asset": 20, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.6, "weight_price": 0.1, "weight_sub_class": 0.3},
                  {"knn_asset": 10, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0, "weight_sub_class": 0.6},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0, "weight_price": 0.5, "weight_sub_class": 0.5},
                  {"knn_asset": 100, "knn_price": 50, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.1, "weight_sub_class": 0.8},
                  {"knn_asset": 100, "knn_price": 80, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.2, "weight_sub_class": 0.7},
                  {"knn_asset": 100, "knn_price": 30, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.7, "weight_sub_class": 0},

                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 10, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.5, "weight_sub_class": 0.1},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 20, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.3, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 30, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.2, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 40, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0.1, "weight_sub_class": 0.5},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 50, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0, "weight_sub_class": 0.6},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 60, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.5, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 70, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.3, "weight_sub_class": 0.4},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 80, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.5, "weight_price": 0.2, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 90, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.6, "weight_price": 0.1, "weight_sub_class": 0.3},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 0, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.4, "weight_price": 0, "weight_sub_class": 0.6},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0, "weight_price": 0.5, "weight_sub_class": 0.5},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.1, "weight_sub_class": 0.8},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.1, "weight_price": 0.2, "weight_sub_class": 0.7},
                  {"knn_asset": 100, "knn_price": 100, "knn_sub_class": 100, "shrink_asset": 2, "shrink_price": 2,
                   "shrink_sub_class": 2, "weight_asset": 0.3, "weight_price": 0.7, "weight_sub_class": 0}
                  ]

    print(cbf_recommender.find_best_fit(parameters))
    # RunRecommender.run_test_recommender(cbf_recommender)


    # map_10 = RunRecommender.run_test_recommender(cbf_recommender)
    # print(map_10)
    # RunRecommender.run(cbf_recommender)
