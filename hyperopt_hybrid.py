import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from Hybrid_CF_CBF import HybridUCFICFRecommender
from Hybrid_ElasticNet_ICF import HybridElasticNetICFUCF
from Hybrid_CF_CBF_ElasticNet import HybridUCFICFElasticNetRecommender
from utils.run import RunRecommender
from utils.helper import Helper

# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)
hybrid = HybridElasticNetICFUCF(Helper().URM_train_validation, mode="validation")

# Step 1 : defining the objective function
def objective(params):
    print(params)
    loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss


# step 2 : defining the search space
search_space = {
    'user_cf_weight': hp.hp.uniform('user_cf_weight', 0.0, 1.0),
    'item_cf_weight': hp.hp.uniform('item_cf_weight', 0.0, 1.0),
    #'user_cbf': hp.hp.uniform('user_cbf', 0.0, 0.5),
    #'item_cbf': hp.hp.uniform('item_cbf', 0.0, 0.5),
    'SLIM_weight': hp.hp.uniform('SLIM_weight', 0.0, 1.0)}

# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 50

#best_submission = {"user_cf": 0.03, "item_cf": 0.88, "user_cbf": 0.07, "item_cbf": 0.02}
#best_submission = {'item_cbf': 0.10488998974023717, 'item_cf': 0.17954977518976983, 'slim_elastic': 0.9760537879950291, 'user_cbf': 0.3089215338230326, 'user_cf': 0.015905291203392107}
old_best = {'SLIM_weight': 0.7599599604986153, 'item_cf_weight': 0.5276012063621809, 'user_cf_weight': 0.03345075223734528}
new_best = {'SLIM_weight': 0.6719606935709935, 'item_cf_weight': 0.08340610330630326,
           'user_cf_weight': 0.006456181309865703}

# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[old_best, new_best])

# best will the return the the best hyperparameter set

print(best)

RunRecommender.evaluate_on_test_set(HybridElasticNetICFUCF, best)
