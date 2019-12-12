import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from Hybrid_ALS_CF import HybridALSCF
from utils.run import RunRecommender
from utils.helper import Helper

# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)
hybrid = HybridALSCF(Helper().URM_train_validation, mode="validation")

# Step 1 : defining the objective function
def objective(params):
    print(params)
    loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss


# step 2 : defining the search space
search_space = {
    'als_weight': hp.hp.uniform('als_weight', 0.0, 1.0),
    'user_cf_weight': hp.hp.uniform('user_cf_weight', 0.0, 1.0),
    'item_cf_weight': hp.hp.uniform('item_cf_weight', 0.0, 1.0)
}

# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 50



# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

# best will the return the the best hyperparameter set

print(best)

RunRecommender.evaluate_on_test_set(hybrid, best)
