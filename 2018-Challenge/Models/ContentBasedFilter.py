from Legacy.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
# from RecommenderSystem import RecommenderSystem
import numpy as np

class ContentBasedFilter(object):
    def __init__(self, URM_tuples, ICM):
        
        self.URM_csr = URM_tuples

        # self.URM = URM
        self.ICM = ICM

        if self.ICM is not list:
            self.ICM = [self.ICM]

        self.W_sparse = []
      
    def fit(self, topK=50, shrink=100, normalize = True, similarity = "cosine"):
        #TODO Verify if there is more than one feature!

        for ICM in self.ICM:
            similarity_object = Compute_Similarity_Python(ICM.T, shrink=shrink, 
                                                  topK=topK, normalize=normalize, 
                                                  similarity = similarity)
        
        self.W_sparse.append(similarity_object.compute_similarity())

    def compute_scores(self, user_id):
        weights = [0.45, 0.55]
        user_profile = self.URM_csr[user_id]
        scores = np.zeros_like(user_profile)
        for W, weight in zip(self.W_sparse, weights):
            scores = scores + weight * user_profile.dot(W).toarray().ravel()

        return scores

        #scores = user_profile.dot(self.W_sparse).toarray().ravel()
        

        
    def recommend(self, user_id, at=10, exclude_seen=True):
        # compute the scores using the dot product
        
        scores = self.compute_scores(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
            
        return ranking[:at]
    
    
    def filter_seen(self, user_id, scores):

        start_pos = self.URM_csr.indptr[user_id]
        end_pos = self.URM_csr.indptr[user_id+1]

        user_profile = self.URM_csr.indices[start_pos:end_pos]
        
        scores[user_profile] = -np.inf

        return scores