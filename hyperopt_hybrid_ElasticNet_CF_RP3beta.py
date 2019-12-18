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
    'SLIM_weight': hp.hp.uniform('SLIM_weight', 0.0, 1.0),
    'user_cf_weight': hp.hp.uniform('user_cf_weight', 0.0, 1.0),
    'item_cf_weight': hp.hp.uniform('item_cf_weight', 0.0, 1.0),
    'rp3_weight': hp.hp.uniform('rp3_weight', 0.0, 1.0)
}


# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 100

# old_best = {'SLIM_weight': 0.6719606935709935, 'item_cf_weight': 0.08340610330630326, 'user_cf_weight': 0.006456181309865703}
# new_best = {'SLIM_weight': 0.758796302775681, 'item_cf_weight': 0.22014675773559156, 'user_cf_weight': 0.011064696091994793}

opt = {'SLIM_weight': 0.6719606935709935, 'item_cf_weight': 0.08340610330630326, 'user_cf_weight': 0.006456181309865703, 'rp3_weight': 0.0}
opt2 =  {'SLIM_weight': 0.8048223770755749, 'item_cf_weight': 0.21238894307574496, 'rp3_weight': 0.12587087079797205, 'user_cf_weight': 0.040353203361717556}

# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt)

# best will the return the the best hyperparameter set

print(best)

RunRecommender.evaluate_on_test_set(HybridElasticNetICFUCFRP3Beta, best)
