import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK, space_eval
from Hybrid_ElasticNet_CF_RP3beta import HybridElasticNetICFUCFRP3Beta
from utils.run import RunRecommender
from utils.helper import Helper

# HybridUCFICFRecommender = HybridUCFICFRecommender(Helper().URM_train_test)

N_KFOLD = 10

kfold = True

if kfold:
    print("Getting Kfold splits...\n")
    kfold_data = Helper().get_kfold_data(N_KFOLD)
    print("Done!\n")
    recommender_list = []
    for i in range(N_KFOLD):
        print("Fitting recommender", i+1, "...\n")
        recommender_list.append(HybridElasticNetICFUCFRP3Beta(kfold_data[i][0], mode="validation"))
        print("\nCompleted!\n")
else:
    hybrid = HybridElasticNetICFUCFRP3Beta(Helper().URM_train_validation, mode="validation")

# Step 1 : defining the objective function
def objective(params):
    print("\n############## New iteration ##############\n", params)
    if kfold:
        loss = - RunRecommender.evaluate_hybrid_weights_validation_kfold(recommender_list, params, kfold=N_KFOLD, parallelize_evaluation=True)
    else:
        loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss


# step 2 : defining the search space
search_space = {
    'SLIM_weight': hp.hp.uniform('SLIM_weight', 0.6, 1),
    'item_cbf_weight': hp.hp.uniform('item_cbf_weight', 0.01, 0.1),
    'item_cf_weight': hp.hp.uniform('item_cf_weight', 0.001, 0.01),
    'rp3_weight': hp.hp.uniform('rp3_weight', 0.6, 1)
}


# step 3 : storing the results of every iteration
bayes_trials = Trials()
MAX_EVALS = 100


opt = {'SLIM_weight': 0.8950096358670148, 'item_cbf_weight': 0.034234727663263104, 'item_cf_weight': 0.011497379340447589, 'rp3_weight': 0.8894480634395567}

new_opt = {'SLIM_weight': 0.8525330515257261, 'item_cbf_weight': 0.03013686377319209, 'item_cf_weight': 0.01129668459365759, 'rp3_weight': 0.9360587800999112}

last_opt = {'SLIM_weight': 0.8737840927419455, 'item_cbf_weight': 0.037666643326618406, 'item_cf_weight': 0.014294955186782246, 'rp3_weight': 0.9314974601074552}



# Optimize
best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[opt, new_opt, last_opt])

best = space_eval(search_space, best)

# best will the return the the best hyperparameter set

print("\n############## Best Parameters ##############\n")
print(best, "\n\nEvaluating on test set now...")

RunRecommender.evaluate_on_test_set(HybridElasticNetICFUCFRP3Beta, best, Kfold=N_KFOLD)


####################################################
# Test Map to beat 0.05112213282549001             #
# MAP-10 score: 0.05107393648464094 on kfold, k = 4#
####################################################

# {'SLIM_weight': 0.7989266787188458, 'item_cbf_weight': 0.03258554983815878, 'item_cf_weight': 0.0077609799300920445, 'rp3_weight': 0.6740989817682256}