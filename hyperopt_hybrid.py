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
    print(params)
    loss = - RunRecommender.evaluate_hybrid_weights_test(HybridElasticNetICF, params)
    return loss


# step 2 : defining the search space
search_space = {
    #'user_cf_weight': hp.hp.uniform('user_cf', 0.0, 0.5),
    #'item_cf_weight': hp.hp.uniform('item_cf', 0.0, 1.0),
    #'user_cbf_weight': hp.hp.uniform('user_cbf', 0.0, 0.5),
    #'item_cbf_weight': hp.hp.uniform('item_cbf', 0.0, 0.5)
    'SLIM_weight': hp.hp.uniform('SLIM', 0.0, 1.0),
    'random_state': hp.hp.choice('random_state', [1234])}

# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 20

#best_submission = {"user_cf": 0.03, "item_cf": 0.88, "user_cbf": 0.07, "item_cbf": 0.02}
best_submission = {'SLIM': 0.9359935612458622, 'item_cf': 0.02700546735599607}

# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[best_submission])

# best will the return the the best hyperparameter set

print(best)
