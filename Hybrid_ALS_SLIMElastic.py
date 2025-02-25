from ALS import AlternatingLeastSquare
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from TopPopularRecommender import TopPopRecommender


import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper



SLIM_parameters = {'alpha': 0.0023512567548654, 'l1_ratio': 0.0004093694334328875, 'positive_only': True, 'topK': 25}
als_parameters = {'n_factors': 680, 'regularization': 0.08905996811679368, 'iterations': 30}


class HybridALSElasticNet(object):

    RECOMMENDER_NAME = "HybridElasticNetICF"

    def __init__(self, URM_train, mode="validation"):
        self.URM_train = URM_train



        # Init single recommenders
        self.top_pop = TopPopRecommender(URM_train)
        self.SLIM = MultiThreadSLIM_ElasticNet(URM_train)
        self.als = AlternatingLeastSquare(URM_train)
        # Get the cold users list
        self.cold_users, _ = Helper().compute_cold_warm_user_ids(URM_train)

        # Fit the single recommenders, this saves time in case of changing just the relative weights

        self.top_pop.fit()
        self.SLIM.fit(**SLIM_parameters)
        self.als.fit(**als_parameters)

    def fit(self, SLIM_weight=0.5):


        self.weights = {"SLIM": SLIM_weight,
                        "als": 1-SLIM_weight}

    def compute_scores(self, user_id):
        scores_als = self.als.compute_scores(user_id)
        scores_SLIM = self.SLIM._compute_item_score(user_id).squeeze()

        scores = (self.weights["als"] * scores_als) + \
                 (self.weights["SLIM"] * scores_SLIM)
        return scores

    def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True):
        if user_id in self.cold_users and enable_toppop:
            # If the user has no interactions a TopPopular recommendation is still better than random
            recommended_items = self.top_pop.recommend(user_id)
        else:
            # Otherwise compute regular recommendations
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

    weights = {'SLIM_weight': 0.9852122782782439}

    hybrid_als_slim = HybridALSElasticNet

    # Evaluation is performed by RunRecommender
    # RunRecommender.evaluate_on_validation_set(hybrid_als_slim, weights)

    RunRecommender.run(hybrid_als_slim, weights)

    from SSLIMElasticNetRecommender import MultiThreadSSLIM_ElasticNet
    from Hybrid_User_CBF_Regional_TopPop import HybridUserCBFRegionalTopPop
    from UserCollaborativeFilter import UserCollaborativeFilter
    from UserCBF import UserCBF

    import numpy as np
    from utils.run import RunRecommender
    from utils.helper import Helper

    UCF_parameters = {'bm_25_norm': True, 'normalize': False, 'shrink': 29, 'similarity': 'cosine', 'topK': 950}

    SSLIM_parameters = {'alpha': 0.005965676587258601, 'bm_25_all': False, 'bm_25_icm': False, 'bm_25_urm': False,
                        'l1_ratio': 0.00024351430967307788, 'positive_only': True, 'side_alpha': 3.6032767753555603,
                        'topK': 20}


    class HybridSSLIMUCF(object):

        RECOMMENDER_NAME = "HybridElasticNetICF"

        def __init__(self, URM_train, mode="validation"):
            self.URM_train = URM_train

            # Init single recommenders
            self.top_pop = HybridUserCBFRegionalTopPop(URM_train)
            self.SSLIM = MultiThreadSSLIM_ElasticNet(URM_train)
            self.UCF = UserCollaborativeFilter(URM_train)
            self.user_cbf = UserCBF(URM_train)
            # Get the cold users list
            self.cold_users, _ = Helper().compute_cold_warm_user_ids(URM_train)

            # Fit the single recommenders, this saves time in case of changing just the relative weights

            self.top_pop.fit()
            self.SSLIM.fit(**SSLIM_parameters)
            self.UCF.fit(**UCF_parameters)
            self.user_cbf.fit(**user_cbf_parameters)

        def fit(self, SSLIM_weight=0.5):

            self.weights = {"SLIM": SSLIM_weight,
                            "UCF": 1 - SSLIM_weight}

        def compute_scores(self, user_id):
            scores_SLIM = self.SSLIM._compute_item_score(user_id).squeeze()
            scores_UCF = self.UCF.compute_scores(user_id)

            scores = (self.weights["UCF"] * scores_UCF) + \
                     (self.weights["SLIM"] * scores_SLIM)
            return scores

        def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True):
            if user_id in self.cold_users and enable_toppop:
                # If the user has no interactions a TopPopular recommendation is still better than random
                recommended_items = self.top_pop.recommend(user_id)
            else:
                # Otherwise compute regular recommendations
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

        weights = {'SLIM_weight': 0.9852122782782439}

        hybrid_als_slim = HybridSSLIMUCF

        # Evaluation is performed by RunRecommender
        # RunRecommender.evaluate_on_validation_set(hybrid_als_slim, weights)

        RunRecommender.run(hybrid_als_slim, weights)