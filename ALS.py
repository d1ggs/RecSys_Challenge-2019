import numpy as np
import implicit


from utils.run import RunRecommender


class AlternatingLeastSquare:
    """
    Implicit ALS implementation. Reference:
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    """

    def __init__(self, URM_train):
        self.URM_train = URM_train
        

    def fit(self,  n_factors=300, regularization=0.15, iterations=30):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

        sparse_item_user = self.URM_train.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization, iterations=self.iterations)


        alpha_val = 24
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors


    def compute_scores(self, user_id):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)

        return np.squeeze(scores)

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = self.compute_scores(user_id)
        if exclude_seen:
            self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[0: at]

    def filter_seen(self, user_id, scores):
        """Remove items that are in the user profile from recommendations

        :param user_id: array of user ids for which to compute recommendations
        :param scores: array containing the scores for each object"""

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
if __name__ == "__main__":
    # Train and test data are now loaded by the helper

    hybrid_ucficf = AlternatingLeastSquare

    opt_weight = {'n_factors': 140, 'regularization': 0.06160137395385561}


    # Evaluation is performed by RunRecommender
    RunRecommender.evaluate_on_test_set(hybrid_ucficf, opt_weight)
