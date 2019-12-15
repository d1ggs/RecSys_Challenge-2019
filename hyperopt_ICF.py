import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from ItemCollaborativeFilter import ItemCollaborativeFilter
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()
import gc


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(ItemCollaborativeFilter, params)
    return loss

als_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 100, 1)),

    "shrink": hp.hp.choice('shrink', np.arange(0, 50, 1))
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    opt = {"topK": 5, "shrink": 8}

    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt)

    ### best will the return the the best hyperparameter set

    print(best)