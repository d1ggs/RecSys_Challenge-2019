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


class ItemBasedCBF:

    def __init__(self, topK_sub_class=200,shrink_sub_class=0):

        self.topK_sub_class = topK_sub_class
        self.shrink_sub_class = shrink_sub_class
        self.helper = Helper()

    def compute_similarity_cbf(self, ICM, top_k, shrink, normalize=True, similarity="cosine"):
        # Compute similarities for weighted features recommender
        similarity_object = Compute_Similarity_Python(ICM.T, shrink=shrink, topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        w_sparse = similarity_object.compute_similarity()
        return w_sparse

    def fit(self, URM):

        # URM Loading
        self.URM_train = URM

        # Load ICMs from helper
        self.ICM_sub_class = self.helper.load_icm_sub_class()
        # Computing SMs
        self.SM_sub_class = self.compute_similarity_cbf(self.ICM_sub_class, top_k=self.topK_sub_class, shrink=self.shrink_sub_class)

    def compute_scores(self, user_id):
        users_list_train = self.URM_train[user_id]
        scores_sub_class = users_list_train.dot(self.SM_sub_class).toarray().ravel()

        return scores_sub_class

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

    def find_best_fit(self, parameter_list):
        print("Looking for best parameter set over ", len(parameter_list))
        results = []
        best = 0
        best_parameters = None
        best_model = None

        # Collect MAP scores for each parameter set

        for parameter_set in tqdm(parameter_list):
            self.__init__(topK_asset=parameter_set["topK_asset"], topK_price=parameter_set["topK_price"],
                          topK_sub_class=parameter_set["topK_sub_class"], shrink_asset=parameter_set["shrink_asset"],
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

    # evaluator.split_data_randomly()

    parameters = { "topK_sub_class": 800,
                   "shrink_sub_class": 100}
    cbf_recommender = ItemBasedCBF(topK_sub_class=parameters["topK_sub_class"], shrink_sub_class=parameters["shrink_sub_class"])

    RunRecommender.perform_evaluation(cbf_recommender)
    # RunRecommender.run(cbf_recommender)
