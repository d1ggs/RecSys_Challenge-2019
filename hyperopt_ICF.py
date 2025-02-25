import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from ItemCollaborativeFilter import ItemCollaborativeFilter
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()

N_K_FOLD = 10

### Step 1 : defining the objective function
def objective(params):
    print('###########################################')
    params["topK"] = int(params["topK"])
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(ItemCollaborativeFilter, params, Kfold=N_K_FOLD, parallel_fit=False, parallelize_evaluation=True)
    return loss

search_space = {
    "topK": hp.hp.quniform('topK', 0, 1000, 5),
    "shrink": hp.hp.uniformint('shrink', 0, 50),
    "bm_25_norm": hp.hp.choice('bm_25_norm', [True, False]),
    "normalize": hp.hp.choice('normalize', [True, False]),
    "similarity": hp.hp.choice('similarity', ["cosine", "jaccard", "dice"])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    item_cf_parameters = {'shrink': 46.0, 'similarity': "jaccard", 'topK': 8}

    # Optimize
    best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=item_cf_parameters)

    params = space_eval(search_space, best)

    ### best will the return the the best hyperparameter set


    print("Best parameters:")
    print(params)
    params["topK"] = int(params["topK"])


    RunRecommender.evaluate_on_test_set(ItemCollaborativeFilter, best, parallel_fit=False, Kfold=N_K_FOLD, parallelize_evaluation=True )
