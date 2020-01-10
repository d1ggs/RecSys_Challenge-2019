import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from utils.run import RunRecommender
from UserBasedCBF import UserBasedCBF
import numpy as np

N_FOLD = 10

### Step 1 : defining the objective function
def objective(params):
    print('###########################################')
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(UserBasedCBF, params, Kfold=N_FOLD, parallel_fit=False, parallelize_evaluation=True, user_group="warm")
    return loss

item_cbf_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 500, 5)),
    "shrink": hp.hp.uniformint('shrink', 0, 50),
    "similarity": hp.hp.choice('similarity', ["cosine", "jaccard", "dice"]),
    "bm_25_normalization": hp.hp.choice('bm_25_normalization', [True, False]),

    "normalize": hp.hp.choice('normalize', [True, False])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    # Optimize
    best = fmin(fn=objective, space=item_cbf_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    params = space_eval(item_cbf_space, best)

    print('################### Best parameters ######################')
    print("Parameters:", params)

    print('################### Performance on Test set ######################')
    RunRecommender.evaluate_on_test_set(UserBasedCBF, params, Kfold=N_FOLD, parallelize_evaluation=True, user_group="warm")
