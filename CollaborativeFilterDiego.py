from base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from utils.helper import Helper
from utils.run import RunRecommender

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

    helper = Helper()
    cb = CollaborativeFilter()

    map10 = RunRecommender.run_test_recommender2(cb)

    cb = CollaborativeFilter()

    map10 = RunRecommender.run_test_recommender2(cb)
    cb = CollaborativeFilter()

    map10 = RunRecommender.run_test_recommender2(cb)
    #print('{0:.128f}'.format(map10))
    print(map10)
