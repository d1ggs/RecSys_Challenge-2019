import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from ItemCollaborativeFilter import ItemCollaborativeFilter
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()

### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(ItemCollaborativeFilter, params, Kfold=4, parallel_fit=True)
    return loss

search_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 100, 5)),
    "shrink": hp.hp.uniformint('shrink', 0, 50),
    "bm_25_norm": hp.hp.choice('bm_25_norm', [True, False]),
    "similarity": hp.hp.choice('similarity', ["cosine", "jaccard", "dice", "tversky", "tanimoto"])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    opt = {"topK": 5, "shrink": 8}

    # Optimize
    best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt)

    best = space_eval(search_space, best)

    ### best will the return the the best hyperparameter set

    print(best)

    RunRecommender.evaluate_on_test_set(ItemCollaborativeFilter, best, parallel_fit=True, Kfold=4, )
