import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from evaluation.Evaluator import Evaluator
import numpy as np
from xgbooster import XGBooster
from utils.helper import Helper
from Hybrid_ElasticNet_ICF_UCF import HybridElasticNetICFUCF
from UserCollaborativeFilter import UserCollaborativeFilter
from copy import deepcopy

### Step 1 : defining the objective function
def objective(params):
    print(params)
    total_loss = 0
    for k in range(4):

        URM_train, URM_test, validation_data, test_data = Helper().get_kfold_data(4)[k]

        booster = XGBooster(URM_train, validation_data, HybridElasticNetICFUCF)

        booster.URM_test = URM_test

        booster.fit(train_parameters=deepcopy(params))
        loss, _ = Evaluator(test_mode=True).evaluate_recommender_kfold(booster, test_data, sequential=True)
        total_loss += loss

    total_loss /= 4

    print("Map@10 k-fold score:", total_loss)
    return -total_loss


xgboost_space = {
    'max_depth': hp.hp.choice('max_depth', np.arange(3, 10)),  # the maximum depth of each tree
    'eta': hp.hp.qloguniform('eta', np.log(1e-3), np.log(0.3), 0.0005),  # step for each iteration
    'num_round': hp.hp.choice('num_round', np.arange(10, 350, 10)),
    'alpha': hp.hp.quniform('alpha', 0, 1, 0.05),
    'lambda': hp.hp.quniform('lambda', 0, 1, 0.05),
    #'learning_rate': hp.hp.qloguniform('learning_rate', np.log(1e-5), np.log(1), 0.00005)
}

if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    # Optimize
    best = fmin(fn=objective, space=xgboost_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    print(best)
