import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from Hybrid_CF_CBF_ElasticNet import HybridCFCBFElasticNet
from utils.run import RunRecommender
from utils.helper import Helper

# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)
hybrid = HybridCFCBFElasticNet(Helper().URM_train_validation, mode="validation")

# Step 1 : defining the objective function
def objective(params):
    print(params)
    loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss


# step 2 : defining the search space
search_space = {
    'item_cf_weight': hp.hp.uniform('item_cf_weight', 0.3, 0.45),
    'slim_elastic_weight': hp.hp.uniform('slim_elastic_weight', 0.7, 0.85),
    'item_cbf_weight': hp.hp.uniform('item_cbf_weight', 0.02, 0.045)}

# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 50


opt = {'item_cbf_weight': 0.0381705753809474, 'item_cf_weight': 0.3588187238528404, 'slim_elastic_weight': 0.7995537607731212}

# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt)

# best will the return the the best hyperparameter set

print(best)

RunRecommender.evaluate_on_test_set(HybridCFCBFElasticNet, best)

"""
MAP-10 score to beat on test: 0.05103293992947268
"""