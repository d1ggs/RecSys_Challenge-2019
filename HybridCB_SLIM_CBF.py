from ItemBasedCBF import ItemBasedCBF
from CollaborativeFilter import CollaborativeFilter
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from TopPopularRecommender import TopPopRecommender
import numpy as np

from utils.helper import Helper
from utils.run import RunRecommender

item_cbf_parameters = {"topK_asset": 100,
                       "topK_price": 100,
                       "topK_sub_class": 200,
                       "shrink_asset": 2,
                       "shrink_price": 2,
                       "shrink_sub_class": 0,
                       "weight_asset": 0.2,
                       "weight_price": 0.15,
                       "weight_sub_class": 0.65}
cb_parameters = {"topK": 210,
                 "shrink": 0}
slim_parameters = {"lr": 0.1, "epochs": 50, "top_k": 20}


class HybridCBCBFSLIMRecommender():
    def __init__(self, weights):
        self.weights = weights
        self.cbf = ItemBasedCBF(topK_asset=item_cbf_parameters["topK_asset"], topK_price=item_cbf_parameters["topK_price"],
                                topK_sub_class=item_cbf_parameters["topK_sub_class"], shrink_asset=item_cbf_parameters["shrink_asset"],
                                shrink_price=item_cbf_parameters["shrink_price"], shrink_sub_class=item_cbf_parameters["shrink_sub_class"],
                                weight_asset=item_cbf_parameters["weight_asset"], weight_price=item_cbf_parameters["weight_price"],
                                weight_sub_class=item_cbf_parameters["weight_sub_class"])
        self.cb = CollaborativeFilter(topK=cb_parameters["topK"], shrink=cb_parameters["shrink"])
        self.toppop = TopPopRecommender()
        self.cold_users = Helper().get_cold_user_ids()

    def set_weights(self, weights):
        self.weights = weights


    def fit(self, URM_train):
        self.URM_train = URM_train
        self.slim = SLIM_BPR_Cython(self.URM_train)

        # Fit the URM into single recommenders
        self.slim.fit(epochs=slim_parameters["epochs"], learning_rate=slim_parameters["lr"], topK=slim_parameters["top_k"])
        self.toppop.fit(self.URM_train)
        self.cbf.fit(self.URM_train)
        self.cb.fit(self.URM_train)

    def compute_scores(self, user_id):

        if user_id in self.cold_users:
            # If the user has no interactions a TopPopular recommendation is still better than random
            scores = self.toppop.recommend(user_id)
        else:
            # Otherwise proceed with regular recommendations
            scores_cbf = self.cbf.compute_scores(user_id)
            scores_cb = self.cb.compute_scores(user_id)
            scores_slim = self.slim._compute_item_score(user_id_array=np.asarray(user_id))
            scores = (self.weights["cbf"] * scores_cbf) + (self.weights["cb"] * scores_cb) + (self.weights["slim"] * scores_slim.squeeze())

        return scores


    def recommend(self, user_id, at=10, exclude_seen=True):
        scores = self.compute_scores(user_id)
        if exclude_seen:
            self.filter_seen(user_id, scores)
        recommended_items = np.argsort(scores)
        recommended_items = np.flip(recommended_items, axis=0)
        return recommended_items[0 : at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


if __name__ == "__main__":

    # Train and test data are now loaded by the helper
    map = []
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf = HybridCBCBFSLIMRecommender(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0.1, "cb": 0.4, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)

    # Evaluation is performed by RunRecommender
    map = map.append(RunRecommender.perform_evaluation(hybrid_cbcbf))
    weights_hybrid = {"cbf": 0, "cb": 0.5, "slim": 0.5}

    hybrid_cbcbf.set_weights(weights_hybrid)