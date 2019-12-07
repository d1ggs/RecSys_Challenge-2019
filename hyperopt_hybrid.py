import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from Hybrid_CF_CBF import HybridUCFICFRecommender
from Hybrid_ElasticNet_ICF import HybridElasticNetICF
from utils.run import RunRecommender
from utils.helper import Helper

# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)
HybridElasticNetICF = HybridElasticNetICF(Helper().URM_train_test, mode="test")

# Step 1 : defining the objective function
def objective(params):
    loss = - RunRecommender.evaluate_hybrid_weights_test(HybridElasticNetICF, params)
    return loss


# step 2 : defining the search space
xgb_space = {
    #    max_depth : maximum depth allowed for every tree
    # hp.choice.choice will select 1 value from the given list
    # 'topK': hp.hp.choice('topK', np.arange(0, 1000, 10, dtype=int)),
    #    subsample : maximum allowed rows for every tree
    # 'shrink': hp.hp.choice('shrink', np.arange(0, 1000, 1, dtype=int))
    # hp.hp.quniform returns a float between a given range
    # 'colsample_bytree': hp.hp.quniform('colsample_bytree', 0.5, 1.0, 0.05),
    #    min_child-weight : minimum number of instances required in each node
    # 'min_child_weight': hp.hp.quniform('min_child_weight', 100, 1000, 100),
    #    reg_alpha : L1 regularisation term on weights
    'user_cf_weight': hp.hp.uniform('user_cf', 0.0, 0.5),
    #    reg_lambda : L2 regularisation term on weights
    'item_cf_weight': hp.hp.uniform('item_cf', 0.5, 1.0),
    'user_cbf_weight': hp.hp.uniform('user_cbf', 0.0, 0.5),
    'item_cbf_weight': hp.hp.uniform('item_cbf', 0.0, 0.5)
}

# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 20

best_submission = {"user_cf": 0.03, "item_cf": 0.88, "user_cbf": 0.07, "item_cbf": 0.02}

# Optimize
best = fmin(fn=objective, space=xgb_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[best_submission])

# best will the return the the best hyperparameter set

print(best)
