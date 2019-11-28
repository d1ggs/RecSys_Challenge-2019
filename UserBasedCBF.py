from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
from utils.run import RunRecommender


class UserBasedCBF():

    def __init__(self, topK_age=500, topK_region=100, shrink_age=2, shrink_region=2, age_weight=0.4, normalize=True,
                 similarity="cosine"):
        self.URM_train = None
        self.W_sparse = None
        self.topK_age = topK_age
        self.topK_region = topK_region
        self.shrink_age = shrink_age
        self.shrink_region = shrink_region
        self.normalize = normalize
        self.similarity = similarity
        self.helper = Helper()
        self.age_weight = age_weight

    def compute_similarity(self, UCM, topK, shrink):
        similarity_object = Compute_Similarity_Python(UCM.T, shrink=shrink, topK=topK,
                                                      normalize=True, similarity="cosine")
        return similarity_object.compute_similarity()

    def fit(self, URM_train):
        self.URM_train = URM_train
        # UCMs loading
        self.UCM_age = self.helper.load_ucm_age()
        self.UCM_region = self.helper.load_ucm_region()
        self.UCM_region = self.helper.sklearn_normalization(self.UCM_region, axis=0)

        # Compute similarities
        self.SM_age = self.compute_similarity(self.UCM_age, self.topK_age, self.shrink_age)
        self.SM_region = self.compute_similarity(self.UCM_region, self.topK_region, self.shrink_region)

    def compute_scores(self, user_id):
        users_list_train = self.URM_train[user_id]
        print(users_list_train.shape, self.SM_age.shape, self.SM_region.shape)
        scores_age = users_list_train.dot(self.SM_age).toarray().ravel()
        scores_region = users_list_train.dot(self.SM_region).toarray().ravel()
        region_weight = 1 - self.age_weight

        scores = (scores_age * self.age_weight) + (scores_region * region_weight)
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
    # evaluator = Evaluator()
    # evaluator.split_data_randomly_2()
    ubcbf = UserBasedCBF()
    ubcbf.helper.split_ucm_region()
    #RunRecommender.perform_evaluation(ubcbf)
    # print('{0:.128f}'.format(map10))
