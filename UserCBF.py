from tqdm import tqdm

from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
from utils.run import RunRecommender
from scipy.sparse import hstack, csc_matrix


def csc_col_set_nz_to_val(csc, column, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csc, csc_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csc.data[csc.indptr[column]:csc.indptr[column + 1]] = value


class UserCBF(object):
    RECOMMENDER_NAME = "UserCBF"

    def __init__(self, URM_train, mode="dataset"):
        self.URM_train = URM_train
        self.W_sparse = None
        self.helper = Helper()
        self.mode = mode
        # UCMs loading
        self.UCM = hstack([self.helper.load_ucm_age(), self.helper.load_ucm_region()])
        self.SM = None
        self.asymmetric_alpha = 0.5

    def get_URM_train(self):
        return self.URM_train

    def compute_similarity(self, UCM):
        similarity_object = Compute_Similarity_Python(UCM.T, shrink=self.shrink, topK=self.topK,
                                                      normalize=self.normalize, similarity=self.similarity,
                                                      asymmetric_alpha=self.asymmetric_alpha)
        return similarity_object.compute_similarity()

    def fit(self, topK=93 * 5, shrink=1, normalize=True, similarity="dice", asymmetric_alpha=0.5, suppress_interactions=True, bm_25_normalization=False):
        self.topK = topK
        self.shrink = shrink
        self.similarity = similarity
        self.asymmetric_alpha = asymmetric_alpha
        self.normalize = normalize

        if bm_25_normalization:
            self.UCM = self.helper.bm25_normalization(self.UCM)

        # Compute similarities
        self.SM = self.compute_similarity(self.UCM)

        # If necessary remove similarity with cold users
        if suppress_interactions:
            cold_users = Helper().get_cold_user_ids(split=self.mode)
            # warm_users = Helper().get_warm_user_ids(split=self.mode)
            self.SM = self.SM.tocsc()

            # Set to zero all similarities with cold users
            for cold in cold_users:
                csc_col_set_nz_to_val(self.SM, cold, 0)

            # And to remove zeros from the sparsity pattern:
            self.SM.eliminate_zeros()

            self.SM.tocsr()

    def compute_scores(self, user_id):
        return self.SM[user_id].dot(self.URM_train).toarray().ravel()

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
    # evaluator = Evaluator()
    # evaluator.split_data_randomly_2()
    ubcbf = UserCBF
    params = {'normalize': True, 'shrink': 1.0, 'similarity': "dice", 'suppress_interactions': True, 'topK': 93 * 5}
    # ubcbf.helper.split_ucm_region()
    RunRecommender.evaluate_on_test_set(ubcbf, params, user_group="cold", parallel_fit=True, Kfold=4)
    # print('{0:.128f}'.format(map10))
