from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
import similaripy as sim
# from evaluation.Evaluator import Evaluator
class ItemCollaborativeFilter(object):

    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.W_sparse = None



    def compute_similarity_matrix(self, URM_train, shrink, topK, normalize, similarity):
        similarity_object = Compute_Similarity_Python(URM_train, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        return similarity_object.compute_similarity()

    def fit(self, topK=500, shrink=2, normalize=True, similarity="cosine"):
        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        self.W_sparse = self.compute_similarity_matrix(self.URM_train, self.shrink, self.topK, self.normalize,
                                                       self.similarity)
        self.URM_W_dot = self.URM_train.dot(self.W_sparse)

    def compute_scores(self, user_id):
        return np.squeeze(np.asarray(self.URM_W_dot[user_id].todense()))

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_scores(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:10]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":
    from utils.run import RunRecommender

    cb_parameters = {'shrink': 17, 'topK': 3}

    cb = ItemCollaborativeFilter

    map10 = RunRecommender.evaluate_on_validation_set(cb, cb_parameters)

