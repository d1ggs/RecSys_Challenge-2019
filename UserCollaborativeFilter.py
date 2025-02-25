from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
from utils.run import RunRecommender
# from evaluation.Evaluator import Evaluator


class UserCollaborativeFilter(object):

    RECOMMENDER_NAME = "UserCollaborativeFilter"

    def __init__(self, URM_train, mode="tomedra"):
        self.URM_train = URM_train
        self.W_sparse = None

    def compute_similarity_matrix(self, URM_train, shrink, topK, normalize, similarity):
        similarity_object = Compute_Similarity_Python(URM_train.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        return similarity_object.compute_similarity()

    def fit(self, topK=410, shrink=0, normalize=True, similarity="cosine", bm_25_norm=False):

        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        if bm_25_norm:
            self.URM_train = Helper().bm25_normalization(self.URM_train)
        self.W_sparse = self.compute_similarity_matrix(self.URM_train, self.shrink, self.topK, self.normalize, self.similarity)

    def compute_scores(self, user_id):
        return self.W_sparse[user_id, :].dot(self.URM_train).toarray().ravel()

    def recommend(self, user_id, at=10, exclude_seen=True):


        # compute the scores using the dot product

        scores = self.compute_scores(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
            
        return ranking[:at]
    
    def filter_seen(self, user_id, scores):

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id+1]

        user_profile = self.URM_train.indices[start_pos:end_pos]
        
        scores[user_profile] = -np.inf

        return scores

if __name__ == "__main__":

    # evaluator = Evaluator()
    # evaluator.split_data_randomly()

    user_cf_parameters = {"topK": 410,
                          "shrink": 0}

    cb = UserCollaborativeFilter

    map10 = RunRecommender.run(cb, user_cf_parameters)
