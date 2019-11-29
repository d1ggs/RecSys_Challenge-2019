from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
from utils.run import RunRecommender

# from evaluation.Evaluator import Evaluator
class ItemCollaborativeFilter(object):

    def __init__(self, topK=500, shrink=2, normalize=True, similarity="cosine"):
        self.URM_train = None
        self.W_sparse = None
        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity

    def compute_similarity_matrix(self, URM_train, shrink, topK, normalize, similarity):
        similarity_object = Compute_Similarity_Python(URM_train, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        return similarity_object.compute_similarity()

    def fit(self, URM_train):
        self.URM_train = URM_train
        self.W_sparse = self.compute_similarity_matrix(self.URM_train, self.shrink, self.topK, self.normalize,
                                                       self.similarity)
        self.URM_W_dot = self.URM_train.dot(self.W_sparse)

    def compute_scores(self, user_id):
        return np.squeeze(np.asarray(self.URM_W_dot[user_id].todense()))

    def recommend(self, user_id, at=10, exclude_seen=False):
        # compute the scores using the dot product

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
    # evaluator = Evaluator()
    # evaluator.split_data_randomly()

    helper = Helper()
    cb_parameters = {"topK": 2,
                     "shrink": 20}
    cb = ItemCollaborativeFilter(topK=cb_parameters["topK"], shrink=cb_parameters["shrink"])

    map10 = RunRecommender.perform_evaluation(cb) # THIS GENERATES A CIRCULAR IMPORT
    # print('{0:.128f}'.format(map10))
