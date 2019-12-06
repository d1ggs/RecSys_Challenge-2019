import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from Hybrid_CF_CBF import HybridUCFICFRecommender
from utils.run import RunRecommender
import numpy as np


### Step 1 : defining the objective function
def objective(params):
    loss = - RunRecommender.evaluate_on_test_set(HybridUCFICFRecommender, params)
    return loss

### step 2 : defining the search space
xgb_space = {
    #    max_depth : maximum depth allowed for every tree
    # hp.choice.choice will select 1 value from the given list
    #'topK': hp.hp.choice('topK', np.arange(0, 1000, 10, dtype=int)),
    #    subsample : maximum allowed rows for every tree
    #'shrink': hp.hp.choice('shrink', np.arange(0, 1000, 1, dtype=int))
    # hp.hp.quniform returns a float between a given range
    #'colsample_bytree': hp.hp.quniform('colsample_bytree', 0.5, 1.0, 0.05),
    #    min_child-weight : minimum number of instances required in each node
    #'min_child_weight': hp.hp.quniform('min_child_weight', 100, 1000, 100),
    #    reg_alpha : L1 regularisation term on weights
    'user_cf_weight': hp.hp.uniform('user_cf', 0.0, 1.0),
    #    reg_lambda : L2 regularisation term on weights
    'item_cf_weight': hp.hp.uniform('item_cf', 0.0, 1.0),
    'user_cbf_weight': hp.hp.uniform('user_cbf', 0.0, 1.0),
    'item_cbf_weight': hp.hp.uniform('item_cbf', 0.0, 1.0)
    }

if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 20

    # Optimize
    best = fmin(fn=objective, space=xgb_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    print(best)