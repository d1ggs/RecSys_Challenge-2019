from ALS import AlternatingLeastSquare
from ItemBasedCBF import ItemCBF
from ItemCollaborativeFilter import ItemCollaborativeFilter
from SSLIMElasticNetRecommender import MultiThreadSSLIM_ElasticNet, SSLIMElasticNetRecommender
from Hybrid_User_CBF_Regional_TopPop import HybridUserCBFRegionalTopPop
from RP3betaRecommender import RP3betaRecommender
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
import numpy as np

from UserCBF import UserCBF
from UserCollaborativeFilter import UserCollaborativeFilter
from utils.run import RunRecommender
from utils.helper import Helper

# item_cf_parameters = {"topK": 29,
#                       "shrink": 22}

# New best parameters
item_cf_parameters = {'shrink': 46.0, 'similarity': "jaccard", 'topK': 8}

als_parameters = {'n_factors': 680, 'regularization': 0.08905996811679368, 'iterations': 30}


rp3_parameters = {'alpha': 0.31932803725825626,
                  'beta': 0.19051435359666555,
                  'implicit': True, 'min_rating': 0,
                  'normalize_similarity': True,
                  'topK': 56}

item_cbf_parameters = {'normalize': True,
                       'shrink': 5,
                       'similarity': 'cosine',
                       'topK': 200}
SLIM_parameters = {'alpha': 0.0023512567548654,
                   'l1_ratio': 0.0004093694334328875,
                   'positive_only': True,
                   'topK': 25}

# SSLIM_parameters_old = {'alpha': 0.0024081648139725204,
#                    'l1_ratio': 0.0007553368138338653,
#                    'positive_only': False,
#                    'side_alpha': 3.86358712510434,
#                    'topK': 65}
#
# SSLIM_parameters = {'alpha': 0.005965676587258601,
#                     'bm_25_all': False,
#                     'bm_25_icm': False,
#                     'bm_25_urm': False,
#                     'l1_ratio': 0.00024351430967307788,
#                     'positive_only': True,
#                     'side_alpha': 3.6032767753555603,
#                     'topK': 20}


# UCF_parameters = {'bm_25_norm': True, 'normalize': False, 'shrink': 29, 'similarity': 'cosine', 'topK': 950}

# user_cbf_parameters = {'bm_25_normalization': False, 'normalize': False, 'shrink': 17, 'similarity': 'jaccard', 'topK': 495}

parameters = {"ItemCollaborativeFilter": item_cf_parameters,
              "SLIMElasticNetRecommender": SLIM_parameters,
              "RP3betaRecommender": rp3_parameters,
              "ItemCBF": item_cbf_parameters,
              "ItemCF": item_cf_parameters,
              "AlternatingLeastSquare": als_parameters,
              "ColdUsersHybrid": {}}


class Hybrid(object):

    RECOMMENDER_NAME = "HybridElasticNetCFRP3Beta"

    def __init__(self, URM_train, recommenders=None, mode="", ):

        if recommenders is None:
            raise AssertionError("Recommenders parameter cannot be None")

        self.URM_train = URM_train
        self.recommenders = {}

        # Initialize and fit the single recommenders
        for recommender in recommenders:
            init_rec = recommender(URM_train)
            init_rec.fit(** parameters[recommender.RECOMMENDER_NAME])
            self.recommenders[recommender.RECOMMENDER_NAME] = init_rec

        # Get the cold users list
        self.cold_users, _ = Helper().compute_cold_warm_user_ids(URM_train)

        # Initialize and fit the hybriod which will deal with cold users
        self.hybrid_cold = HybridUserCBFRegionalTopPop(URM_train)
        self.hybrid_cold.fit()

        # Fit the single recommenders, this saves time in case of changing just the relative weights

    def fit(self, weights: dict):

        weight_sum = 0

        # Normalize the weights, just in case
        for weight in weights.values():
            weight_sum += weight

        for recommender in weights:
            weights[recommender] /= weight_sum

        self.weights = weights


    def compute_scores(self, user_id):
        scores = np.zeros(self.URM_train.shape[1])

        for recommender in self.recommenders:
            # Deal with some legacy implementations
            if recommender == "SLIMElasticNetRecommender" or recommender == "SSLIMElasticNetRecommender" or recommender == "RP3betaRecommender":
                score = self.recommenders[recommender]._compute_item_score(user_id).squeeze()
            # Compute regular scores
            else:
                score = self.recommenders[recommender].compute_scores(user_id)

            # Weight the scores
            scores += score * self.weights[recommender]

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True, enable_toppop=True):
        if user_id in self.cold_users and enable_toppop:
            # If the user has no interactions a TopPopular recommendation is still better than random
            recommended_items = self.hybrid_cold.recommend(user_id)
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
    weights = {"weights": {'AlternatingLeastSquare': 0.09136760425375567, 'ItemCBF': 0.01686781824511765, 'ItemCollaborativeFilter': 0.03454041362675262, 'RP3betaRecommender': 0.8187162817070645, 'SLIMElasticNetRecommender': 0.7756431422518303}}

    RunRecommender.run(Hybrid, weights, init_params={"recommenders": [MultiThreadSLIM_ElasticNet, ItemCollaborativeFilter, RP3betaRecommender, ItemCBF, AlternatingLeastSquare]})

    #RunRecommender.evaluate_on_test_set(hybrid_ucficf, weights, Kfold=10, parallelize_evaluation=True)


