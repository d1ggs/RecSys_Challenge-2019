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

    def __init__(self, URM_train, recommender_class=UserCollaborativeFilter, mode="dataset"):

        self.URM_train = URM_train
        self.validation = Helper().validation_data
        if mode == "dataset":
            self.validation = Helper().test_data
        elif mode == "test":
            self.validation = Helper().validation_data
        else:
            raise AssertionError("Validation mode is not supported by this recommender, use 'test' or 'dataset'")
        self.toppop = TopPopRecommender(URM_train)

        self.recommender = recommender_class(URM_train)

        self.toppop.fit()
        #TODO Check recommender params

        self.recommender.fit()
        self.already_fitted = False

    def train_XGB(self, X_train, y_train, X_test, y_test, params=None, pos_weight=19.0):

        # X_train, X_test, y_train, y_test = train_test_split(X, y)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        deval = xgb.DMatrix(X_test, label=y_test)

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

        early_stopping_rounds = 50
        xgb_model = xgb.train(params,
                              dtrain,
                              num_round,
                              verbose_eval=2,
                              evals=[(dtrain, 'train'), (deval, 'eval')], early_stopping_rounds=early_stopping_rounds)

        return xgb_model

    def generate_dataframe(self, URM, target_data):
        assert target_data is not None, "target_data cannot be None if load_XGB is False"
        # We need to train the model
        user_recommendations_items = []
        user_recommendations_user_id = []

        labels = []

        count = 0

        # Get recommendations from internal recommender model

        for user_id in target_data.keys():
            recommendations = self.recommender.recommend(user_id, at=20)
            assert recommendations.shape[
                       0] == 20, "Internal recommender is not providing the requested recommendation length"
            for recommendation in recommendations:
                if int(recommendation) == target_data[user_id]:
                    labels.append(1)
                    count += 1
                else:
                    labels.append(0)
            user_recommendations_items.extend(recommendations)
            user_recommendations_user_id.extend([user_id] * len(recommendations))

        # Put the item popularity as feature
        topPop_score_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            topPop_score = self.toppop.item_popularity[item_id]
            topPop_score_list.append(topPop_score)

        # Create dataframe

        train_dataframe = pd.DataFrame(
            {"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})

        train_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=train_dataframe.index)

        # Put the user profile length as feature

        user_profile_len = np.ediff1d(URM.indptr)

        user_profile_len_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            user_profile_len_list.append(user_profile_len[user_id])

        train_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=train_dataframe.index)


        pos_weight = (len(user_recommendations_items) - count) / count
        return train_dataframe, pos_weight, labels

    def fit(self, load_XGB=False, target_data=None, train_parameters=None):

        if not load_XGB:
            if not self.already_fitted:
                self.train_dataframe, self.pos_weight_train, self.labels_train = self.generate_dataframe(self.URM_train, self.validation)
                self.already_fitted = True
                self.test_dataframe, _, self.labels_test = self.generate_dataframe(Helper().URM_train_test, Helper().test_data)

            self.XGB_model = self.train_XGB(self.train_dataframe,
                                            self.labels_train,
                                            self.test_dataframe,
                                            self.labels_test,
                                            pos_weight=self.pos_weight_train,
                                            params=train_parameters)

        else:
            raise NotImplementedError("Model loading still needs to be implemented")

    def recommend(self, user_id, at=10, exclude_seen=True):

        # Get internal recommender recommendations
        recommendations = self.recommender.recommend(user_id, at=20, exclude_seen=exclude_seen)

        user_recommendations_items = recommendations
        user_recommendations_user_id = [user_id] * len(recommendations)

        # Put the item popularity as feature
        topPop_score_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            topPop_score = self.toppop.item_popularity[item_id]
            topPop_score_list.append(topPop_score)

        # Build the dataframe

        test_dataframe = pd.DataFrame({"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})
        test_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=test_dataframe.index)

        # Put the user profile length as feature

        user_profile_len = np.ediff1d(self.URM_train.indptr)
        user_profile_len_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            user_profile_len_list.append(user_profile_len[user_id])

        test_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=test_dataframe.index)

        xgb_predictions = self.XGB_model.predict(xgb.DMatrix(test_dataframe))

        indices = np.flip(np.argsort(xgb_predictions))[:at]
        sorted_recommendations = user_recommendations_items[indices]

        assert sorted_recommendations.shape[0] == 10
        return sorted_recommendations

if __name__ == '__main__':
    params={'alpha': 0.9, 'eta': 0.2975, 'lambda': 0.15000000000000002, 'max_depth': 5, 'num_round': 32}

    booster = XGBooster(Helper().URM_train_test)
    booster.fit(load_XGB=False, target_data=Helper().test_data, train_parameters=params)


    # booster = XGBooster(Helper().URM_train_validation)
    # booster.fit(load_XGB=False, target_data=Helper().validation_data)
    # MAP_final, _ = Evaluator(test_mode=True).evaluateRecommender_old(booster, None)

    booster.toppop = TopPopRecommender(Helper().URM_csr)
    booster.user_cf = UserCollaborativeFilter(Helper().URM_csr)
    booster.toppop.fit()
    booster.user_cf.fit()

    RunRecommender.write_submission(booster)


