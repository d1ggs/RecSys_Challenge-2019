import multiprocessing

import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK, space_eval

from ALS import AlternatingLeastSquare
from ItemCollaborativeFilter import ItemCollaborativeFilter
from Hybrid import Hybrid
from Hybrid_SSLIM_CF_RP3beta import HybridSSLIMICFUCFRP3Beta
from ItemBasedCBF import ItemCBF
from RP3betaRecommender import RP3betaRecommender
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from UserCBF import UserCBF
from UserCollaborativeFilter import UserCollaborativeFilter
from utils.run import RunRecommender
from utils.helper import Helper
from utils.schleep import computer_sleep

# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)


recommender_class = Hybrid

recommenders = [MultiThreadSLIM_ElasticNet, ItemCollaborativeFilter, RP3betaRecommender, ItemCBF, AlternatingLeastSquare]

N_KFOLD = 10

kfold = False
parallel_fit = True

if kfold > 0:
    MAP_final = 0
    recommender_list = []

    print("Getting Kfold splits...\n")
    kfold_data = Helper().get_kfold_data(N_KFOLD)
    print("Done!\n")
    for i in range(N_KFOLD):
        print("Fitting recommender", i+1, "...\n")
        recommender_list.append(recommender_class(kfold_data[i][0], recommenders))
        print("\nCompleted!\n")
else:
    hybrid = recommender_class(Helper().URM_train_validation, recommenders)



# Step 1 : defining the objective function
def objective(params):
    print("\n############## New iteration ##############\n", params)
    params = {"weights": params}
    if kfold:
        loss = - RunRecommender.evaluate_hybrid_weights_validation_kfold(recommender_list, params, kfold=N_KFOLD, parallelize_evaluation=kfold, parallel_fit=False)
    else:
        loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss


# step 2 : defining the search space
search_space = {
    'SLIMElasticNetRecommender': hp.hp.uniform('SLIMElasticNetRecommender', 0.75, 1),
    # 'item_cbf_weight': hp.hp.uniform('item_cbf_weight', 0, 0.2),
    'ItemCollaborativeFilter': hp.hp.uniform('ItemCollaborativeFilter', 0.01, 0.04),
    'RP3betaRecommender': hp.hp.uniform('RP3betaRecommender', 0.8, 1),
    # 'UserCBF': hp.hp.quniform('user_cbf_weight', 0, 0.3, 0.0001),
    'ItemCBF': hp.hp.uniform('ItemCollaborative', 0.008, 0.02),
    'AlternatingLeastSquare': hp.hp.uniform('AlternatingLeastSquare', 0.05, 0.2)
}


# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 50


# opt = {'SSLIM_weight': 0.8950096358670148, 'item_cbf_weight': 0.034234727663263104, 'item_cf_weight': 0.011497379340447589, 'rp3_weight': 0.8894480634395567, 'user_cbf_weight': 0, 'user_cf_weight': 0}
#
# new_opt = {'SSLIM_weight': 0.8525330515257261, 'item_cbf_weight': 0.03013686377319209, 'item_cf_weight': 0.01129668459365759, 'rp3_weight': 0.9360587800999112, 'user_cbf_weight': 0, 'user_cf_weight': 0}
#
# last_opt = {'SSLIM_weight': 0.8737840927419455, 'item_cbf_weight': 0.037666643326618406, 'item_cf_weight': 0.014294955186782246, 'rp3_weight': 0.9314974601074552, 'user_cbf_weight': 0, 'user_cf_weight': 0}
#

# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

best = space_eval(search_space, best)

# best will the return the the best hyperparameter set

print("\n############## Best Parameters ##############\n")
print(best, "\n\nEvaluating on test set now...")

RunRecommender.evaluate_on_test_set(Hybrid,  {"weights": best}, Kfold=N_KFOLD)

computer_sleep(verbose=False)


####################################################
# Test Map to beat 0.05112213282549001             #
# MAP-10 score: 0.05107393648464094 on kfold, k = 4#
####################################################

# {'SLIM_weight': 0.7989266787188458, 'item_cbf_weight': 0.03258554983815878, 'item_cf_weight': 0.0077609799300920445, 'rp3_weight': 0.6740989817682256}