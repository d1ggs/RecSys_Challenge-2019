from tqdm import tqdm

from base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np

from evaluation.Evaluator import Evaluator
from utils.helper import Helper
from utils.split_URM import split_train_test


class CollaborativeFilter(object):

    def __init__(self):
        self.URM_train = None
        self.W_sparse = None

    def fit(self, URM_train, topK=500, shrink=2, normalize=True, similarity = "cosine"):
        self.URM_train = URM_train
        similarity_object = Compute_Similarity_Python(self.URM_train.T, shrink=shrink,
                                                  topK=topK, normalize=normalize, 
                                                  similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=10, exclude_seen=False):

        # compute the scores using the dot product
        
        scores = self.W_sparse[user_id, :].dot(self.URM_train).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
            
        return ranking[:10]
    
    def filter_seen(self, user_id, scores):

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id+1]

        user_profile = self.URM_train.indices[start_pos:end_pos]
        
        scores[user_profile] = -np.inf

        return scores

if __name__ == "__main__":

    # evaluator = Evaluator()
    # evaluator.split_data_randomly_2()

    evaluator = Evaluator()
    helper = Helper()

    MAP_final = 0.0
    URM_all = helper.convert_URM_to_csr(helper.URM_data)

    URM_train, URM_test, target_users_test, test_data = split_train_test(URM_all, 0.8)

    recommender = CollaborativeFilter()

    recommender.fit(URM_train)


    for user in tqdm(target_users_test):
        recommended_items = recommender.recommend(int(user), exclude_seen=True)
        relevant_item = test_data[int(user)]

        MAP_final += evaluator.MAP(recommended_items, relevant_item)

    MAP_final /= len(target_users_test)

    print(MAP_final)



    # map10 = RunRecommender.run_test_recommender2(cb)
    #
    # cb = CollaborativeFilter()
    #
    # map10 = RunRecommender.run_test_recommender2(cb)
    # cb = CollaborativeFilter()
    #
    # map10 = RunRecommender.run_test_recommender2(cb)
    # print('{0:.128f}'.format(map10))
    # print(map10)
