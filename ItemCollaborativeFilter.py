from tqdm import tqdm

from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper




class ItemCollaborativeFilter(object):

    def __init__(self, URM_train, mode=""):
        self.URM_train = URM_train
        self.W_sparse = None

    def compute_similarity_matrix(self, URM_train, shrink, topK, normalize, similarity):
        similarity_object = Compute_Similarity_Python(URM_train, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        return similarity_object.compute_similarity()

    def fit(self, topK=10, shrink=46, normalize=True, similarity="jaccard"):
        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity

        if normalize:
            self.URM_train = Helper().bm25_normalization(self.URM_train)

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

    cb_parameters = [{"topK": i} for i in range(1,20)]

    cb = ItemCollaborativeFilter

    max_map = 0

    best_params = None

    for params in tqdm(cb_parameters):
        map10 = RunRecommender.evaluate_on_test_set(cb, params, Kfold=4, sequential_MAP=True)
        if map10 > max_map:
            best_params = params
            max_map = map10

    print("Best map:", max_map)
    print(best_params)

