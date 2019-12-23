import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from Hybrid_ElasticNet_CF_RP3beta import HybridElasticNetICFUCFRP3Beta
from utils.run import RunRecommender
from utils.helper import Helper

# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)
hybrid = HybridElasticNetICFUCFRP3Beta(Helper().URM_train_validation, mode="validation")

# Step 1 : defining the objective function
def objective(params):
    print(params)
    loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss


# step 2 : defining the search space
search_space = {
    'SLIM_weight': hp.hp.uniform('SLIM_weight', 0.85, 0.95),
    'item_cbf_weight': hp.hp.uniform('item_cbf_weight', 0.03, 0.07),
    # 'item_cf_weight': hp.hp.uniform('item_cf_weight', 0, 0.1),
    'item_cf_weight': 0,
    'rp3_weight': hp.hp.uniform('rp3_weight', 0.85, 0.95)
}


# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 100


opt = {'SLIM_weight': 0.8950096358670148, 'item_cbf_weight': 0.034234727663263104, 'item_cf_weight': 0.011497379340447589, 'rp3_weight': 0.8894480634395567}


# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt)

# best will the return the the best hyperparameter set

print(best)

RunRecommender.evaluate_on_test_set(HybridElasticNetICFUCFRP3Beta, best)


##########################################
# Test Map to beat 0.05112213282549001   #
##########################################