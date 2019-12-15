from evaluation.Evaluator import Evaluator
from utils.helper import Helper
import xgboost as xgb
from UserCollaborativeFilter import UserCollaborativeFilter
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from TopPopularRecommender import TopPopRecommender
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.run import RunRecommender


class XGBooster(object):

    def __init__(self, URM_train, recommender_class=UserCollaborativeFilter):

        self.URM_train = URM_train

        self.toppop = TopPopRecommender(URM_train)

        self.recommender = recommender_class(URM_train)

        self.toppop.fit()
        self.recommender.fit()
        self.already_fitted = False

    def train_XGB(self, X, y, params=None, pos_weight=19.0):

        #X_train, X_test, y_train, y_test = train_test_split(X, y)

        dtrain = xgb.DMatrix(X, label=y)
        #dtest = xgb.DMatrix(X_test, label=y_test)

        if params is None:
            params = {
                'max_depth': 5,  # the maximum depth of each tree
                'eta': 0.3,  # step for each iteration
                'silent': 1,  # keep it quiet
                'objective': 'binary:logistic',  # 0/1 output for each sample
                # 'num_class': 2,  # the number of classes
                'eval_metric': 'map',
                'scale_pos_weight': pos_weight}  # evaluation metric
            num_round = 500  # the number of training iterations (number of trees)
        else:
            params['pos_weight'] = pos_weight
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'map'
            params['silent'] = 1
            num_round = params['num_round']
            del params['num_round']


        xgb_model = xgb.train(params,
                              dtrain,
                              num_round,
                              verbose_eval=2,
                              evals=[(dtrain, 'train')])

        return xgb_model

    def fit(self, load_XGB=False, target_data=None, train_parameters=None):

        if not load_XGB:
            if not self.already_fitted:
                self.already_fitted = True
                assert target_data is not None, "target_data cannot be None if load_XGB is False"
                # We need to train the model
                user_recommendations_items = []
                user_recommendations_user_id = []

                self.labels = []

                count = 0

                for user_id in target_data.keys():
                    recommendations = self.recommender.recommend(user_id, at=20)
                    for recommendation in recommendations:
                        if int(recommendation) == target_data[user_id]:
                            self.labels.append(1)
                            count += 1
                        else:
                            self.labels.append(0)
                    user_recommendations_items.extend(recommendations)
                    user_recommendations_user_id.extend([user_id] * len(recommendations))

                # Put the item popularity as feature
                topPop_score_list = []

                for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
                    topPop_score = self.toppop.item_popularity[item_id]
                    topPop_score_list.append(topPop_score)

                train_dataframe = pd.DataFrame(
                    {"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})

                train_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=train_dataframe.index)

                # Put the user profile length as feature

                user_profile_len = np.ediff1d(self.URM_train.indptr)

                user_profile_len_list = []

                for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
                    user_profile_len_list.append(user_profile_len[user_id])

                train_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=train_dataframe.index)

                self.train_dataframe = train_dataframe

                self.pos_weight = (len(user_recommendations_items)-count) / count

            self.XGB_model = self.train_XGB(self.train_dataframe, self.labels, pos_weight=self.pos_weight,
                                            params=train_parameters)

        else:
            raise NotImplementedError("Model loading still needs to be implemented")

    def recommend(self, user_id, at=10, exclude_seen=True):
        user_recommendations_items = []
        user_recommendations_user_id = []

        recommendations = self.recommender.recommend(user_id, at=20, exclude_seen=exclude_seen)

        user_recommendations_items = recommendations
        user_recommendations_user_id = [user_id] * len(recommendations)

        # Put the item popularity as feature
        topPop_score_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            topPop_score = self.toppop.item_popularity[item_id]
            topPop_score_list.append(topPop_score)

        test_dataframe = pd.DataFrame({"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})

        test_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=test_dataframe.index)

        # Put the user profile length as feature

        user_profile_len = np.ediff1d(self.URM_train.indptr)

        user_profile_len_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            user_profile_len_list.append(user_profile_len[user_id])

        test_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=test_dataframe.index)

        xgb_predictions = self.XGB_model.predict(xgb.DMatrix(test_dataframe))

        count = 1
        predictions = []
        predictions_list = []

        indices = np.flip(np.argsort(xgb_predictions))[:at]
        sorted_recommendations = user_recommendations_items[indices]

        # for prediction in xgb_predictions:
        #     predictions_list.append(prediction)
        #     if count % 20 == 0:
        #         predictions.append(np.array(predictions_list))
        #     count += 1
        #
        # for xgb_pred, recommendations in zip(predictions, user_recommendations_items_list):
        #     indices = np.argsort(xgb_pred)[:at]
        #     sorted_recommendations = recommendations[indices]

        assert sorted_recommendations.shape[0] == 10
        return sorted_recommendations

if __name__ == '__main__':
    # booster = XGBooster(Helper().URM_train_validation)
    # booster.fit(load_XGB=False, target_data=Helper().validation_data)

    booster = XGBooster(Helper().URM_train_validation)
    booster.fit(load_XGB=False, target_data=Helper().validation_data)
    MAP_final, _ = Evaluator(test_mode=True).evaluateRecommender_old(booster, None)

    # booster.toppop = TopPopRecommender(Helper().URM_csr)
    # booster.user_cf = UserCollaborativeFilter(Helper().URM_csr)
    # booster.toppop.fit()
    # booster.user_cf.fit()

    # RunRecommender.write_submission(booster)

    print("MAP@10:", MAP_final)

