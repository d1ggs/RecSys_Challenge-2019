import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from utils.run import RunRecommender
from ItemBasedCBF import ItemCBF
import numpy as np


### Step 1 : defining the objective function
def objective(params):
    print('###########################################')
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(ItemCBF, params, Kfold=4, parallel_fit=True)
    return loss

item_cbf_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 500, 5)),
    "shrink": hp.hp.uniformint('shrink', 0, 50),
    "similarity": hp.hp.choice('similarity',
                               ["cosine", "jaccard", "dice", "tversky", "tanimoto"]),
    "bm_25_norm": hp.hp.choice('bm_25_norm', [True, False]),

    "normalize": hp.hp.choice('normalize', [True, False])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    # Optimize
    best = fmin(fn=objective, space=item_cbf_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate={"topK": 1, "shrink": 13, "similarity": "cosine", "normalize": True})

    ### best will the return the the best hyperparameter set

    params = space_eval(item_cbf_space, best)

    print("Best parameters:", params)
    RunRecommender.evaluate_on_test_set(ItemCBF, params, Kfold=4, parallel_fit=True)
