from utils.helper import Helper
import xgboost as xgb
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from TopPopularRecommender import TopPopRecommender
import pandas as pd
import numpy as np


def train(X_train, y_train, X_test, y_test):
    #assert X_train is not None and y_train is not None and X_test is not None and y_test is not None

    dtrain = xgb.DMatrix(X_train, label=y_train)
    #dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # step for each iteration
        'silent': 1,  # keep it quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 3,  # the number of classes
        'eval_metric': 'merror'}  # evaluation metric

    num_round = 20  # the number of training iterations (number of trees)

    model = xgb.train(params,
                      dtrain,
                      num_round,
                      verbose_eval=2,
                      evals=[(dtrain, 'train')])

    return model


cutoff = 20
user_recommendations_items = []
user_recommendations_user_id = []

URM_train = Helper().URM_train_validation
validation_data = Helper().validation_data

toppop = TopPopRecommender(URM_train)
toppop.fit()

slim = MultiThreadSLIM_ElasticNet(URM_train)
slim.fit()

labels = []

for n_user in validation_data.keys():
    recommendations = slim.recommend(n_user, cutoff=20)
    for recommendation in recommendations:
        labels.append(1 if int(recommendation) == validation_data[n_user] else 0)
    user_recommendations_items.extend(recommendations)
    user_recommendations_user_id.extend([n_user] * len(recommendations))

# Put the item popularity as feature
topPop_score_list = []

for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
    topPop_score = toppop.item_popularity[item_id]
    topPop_score_list.append(topPop_score)

train_dataframe = pd.DataFrame({"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})

train_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=train_dataframe.index)

# Put the user profile length as feature

user_profile_len = np.ediff1d(URM_train.indptr)

user_profile_len_list = []

target_feature = 1

for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
    user_profile_len_list.append(user_profile_len[user_id])

train_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=train_dataframe.index)

# TODO Put the item features

model = train(train_dataframe, labels, None, None)
