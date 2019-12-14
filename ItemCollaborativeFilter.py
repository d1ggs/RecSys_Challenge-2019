from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
# from evaluation.Evaluator import Evaluator
class ItemCollaborativeFilter(object):

    def __init__(self, URM_train, mode="validation"):
        self.URM_train = URM_train
        self.W_sparse = None

        self.user_data_dict, self.cluster_dict, self.region_dict, self.age_dict = \
            Helper().get_demographic_info(verbose=False, mode=mode)
        self.cluster_recommendations = {}
        self.region_recommendations = {}
        self.age_recommendations = {}

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

    def recommend(self, user_id, at=10, exclude_seen=False, use_demographic=True):

        if user_id in self.user_data_dict and use_demographic:
            obj = self.user_data_dict[user_id]

            if obj.cluster is not None and obj.age is None and obj.region is None:
                cluster = self.user_data_dict[user_id].cluster
                if cluster in self.cluster_recommendations:
                    scores = self.cluster_recommendations[cluster]
                else:
                    user_list = self.cluster_dict[cluster]
                    scores = np.zeros(shape=self.URM_train.shape[1])
                    for user in user_list:
                        scores += self.compute_scores(user)
                    self.cluster_recommendations[cluster] = scores

            elif obj.cluster is None and obj.age is not None and obj.region is None:
                age = self.user_data_dict[user_id].age
                if age in self.age_recommendations:
                    scores = self.age_recommendations[age]
                else:
                    user_list = self.age_dict[age]
                    scores = np.zeros(shape=self.URM_train.shape[1])
                    for user in user_list:
                        scores += self.compute_scores(user)
                    self.age_recommendations[age] = scores

            elif obj.cluster is None and obj.age is  None and obj.region is not None:
                region = self.user_data_dict[user_id].region
                if region in self.region_recommendations:
                    scores = self.region_recommendations[region]
                else:
                    user_list = self.region_dict[region]
                    scores = np.zeros(shape=self.URM_train.shape[1])
                    for user in user_list:
                        scores += self.compute_scores(user)
                    self.region_recommendations[region] = scores
            else:
                print("Something went wrong")
                exit()

        else:
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
    from utils.run import RunRecommender

    cb_parameters = {"topK": 5,
                     "shrink": 8}
    cb = ItemCollaborativeFilter

    map10 = RunRecommender.evaluate_on_test_set(cb, cb_parameters) # THIS GENERATES A CIRCULAR IMPORT

    print(map10)
    cb_parameters = {"topK": 29,
                     "shrink": 22}
    print(RunRecommender.evaluate_on_test_set(cb, cb_parameters))
    # print('{0:.128f}'.format(map10))
