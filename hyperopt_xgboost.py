import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from evaluation.Evaluator import Evaluator
import numpy as np
from xgbooster import XGBooster
from utils.helper import Helper
from Hybrid_ElasticNet_ICF_UCF import HybridElasticNetICFUCF
from UserCollaborativeFilter import UserCollaborativeFilter

booster = XGBooster(Helper().URM_train_validation, recommender_class=UserCollaborativeFilter)


### Step 1 : defining the objective function
def objective(params):
    print(params)
    booster.fit(target_data=Helper().validation_data, train_parameters=params)
    loss, _ = Evaluator(test_mode=True).evaluateRecommender_old(booster, None)
    print("Map@10 score:", loss)
    return -loss


xgboost_space = {
    'max_depth': hp.hp.choice('max_depth', np.arange(1, 15)),  # the maximum depth of each tree
    'eta': hp.hp.qloguniform('eta', np.log(1e-5), np.log(1), 0.00005),  # step for each iteration
    'num_round': hp.hp.choice('num_round', np.arange(10, 500, 10)),
    'learning_rate': hp.hp.qloguniform('learning_rate', np.log(1e-5), np.log(1), 0.00005)
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
