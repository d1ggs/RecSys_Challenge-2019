from tqdm import tqdm

from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
from utils.run import RunRecommender
from scipy.sparse import hstack, lil_matrix, csc_matrix

def csc_col_set_nz_to_val(csc, column, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csc, csc_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csc.data[csc.indptr[column]:csc.indptr[column+1]] = value


class UserBasedCBF(object):
    RECOMMENDER_NAME = "UserBasedCBF"

    def __init__(self, URM_train, mode="dataset") :
        self.URM_train = URM_train
        self.W_sparse = None
        self.helper = Helper()
        self.mode = mode
        # UCMs loading
        self.UCM = hstack([self.helper.load_ucm_age(), self.helper.load_ucm_region()])
        self.SM = None

    def get_URM_train(self):
        return self.URM_train

    def compute_similarity(self, UCM, topK, shrink):
        similarity_object = Compute_Similarity_Python(UCM.T, shrink=shrink, topK=topK,
                                                      normalize=True, similarity="cosine")
        return similarity_object.compute_similarity()

    def fit(self, topK=300, shrink=1, normalize=True, similarity="cosine", suppress_interactions=True):
        self.topK = topK
        self.shrink = shrink
        self.similarity = similarity

        if normalize:
            self.UCM = self.helper.bm25_normalization(self.UCM)

        # Compute similarities
        self.SM = self.compute_similarity(self.UCM, self.topK, self.shrink)

        if suppress_interactions:
            print("Suppressing similarity with cold users...")
            cold_users = Helper().get_cold_user_ids(split=self.mode)
            # warm_users = Helper().get_warm_user_ids(split=self.mode)
            self.SM = self.SM.tocsc()

            # Now you can just do:
            for cold in cold_users:
                csc_col_set_nz_to_val(self.SM, cold, 0)

            # And to remove zeros from the sparsity pattern:
            self.SM.eliminate_zeros()

            self.SM.tocsr()

            print("Done!")

    def compute_scores(self, user_id):


        return self.SM[user_id].dot(self.URM_train).toarray().ravel()

    def recommend(self, user_id, cutoff=10, exclude_seen=True, remove_top_pop_flag=False,
                  remove_custom_items_flag=False, return_scores = False):
        # Compute scores of the recommendation
        scores = self.compute_scores(user_id)

        # Filter to exclude already seen items
        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[:cutoff]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":
    # evaluator = Evaluator()
    # evaluator.split_data_randomly_2()
    ubcbf = UserBasedCBF
    #ubcbf.helper.split_ucm_region()
    RunRecommender.evaluate_on_test_set(ubcbf, {}, user_group="cold")
    # print('{0:.128f}'.format(map10))
