from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from utils.helper import Helper
from utils.run import RunRecommender
from scipy.sparse import hstack
import numpy as np


class ItemCBF:

    RECOMMENDER_NAME = "ItemCBF"

    def __init__(self, URM, mode="dataset"):

        self.URM_train = URM

        self.helper = Helper()

    def compute_similarity_cbf(self, ICM, top_k, shrink, normalize=True, similarity="asymmetric"):
        # Compute similarities for weighted features recommender
        similarity_object = Compute_Similarity_Python(ICM.T, shrink=shrink, topK=top_k, normalize=normalize,
                                                      similarity=similarity)
        w_sparse = similarity_object.compute_similarity()
        return w_sparse

    def fit(self, topK=15, shrink=18, similarity="asymmetric", normalize=True, bm_25_norm=True):

        self.topK = topK
        self.shrink = shrink
        self.similarity = similarity

        # Load ICMs from helper
        ICM_sub_class = self.helper.load_icm_sub_class()
        ICM_asset = self.helper.load_icm_asset()
        ICM_price = self.helper.load_icm_price()
        matrices = [ICM_asset, ICM_sub_class, ICM_price]
        self.ICM = hstack(matrices)

        if bm_25_norm:
            self.URM_train = Helper().bm25_normalization(self.URM_train)

        # Computing SMs
        self.SM = self.compute_similarity_cbf(self.ICM, top_k=self.topK, shrink=self.shrink, similarity=self.similarity, normalize=normalize)

    def compute_scores(self, user_id):
        users_list_train = self.URM_train[user_id]
        scores_sub_class = users_list_train.dot(self.SM).toarray().ravel()

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


if __name__ == "__main__":

    cbf_recommender = ItemCBF

    params = {'normalize': True, 'shrink': 18, 'similarity': 'asymmetric', 'topK': 15}

    RunRecommender.evaluate_on_validation_set(cbf_recommender, params)
    RunRecommender.evaluate_on_validation_set(cbf_recommender, params)
    # RunRecommender.run(cbf_recommender)
