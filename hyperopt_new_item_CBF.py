import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from AssetCBF import AssetCBF
from utils.run import RunRecommender
from ItemBasedCBF_new import ItemCBF
import numpy as np


### Step 1 : defining the objective function
def objective(params):
    loss = - RunRecommender.evaluate_on_test_set(ItemCBF, params)
    return loss

price_cbf_space = {
    "topK": hp.hp.choice('topK', [0, 20, 80, 100, 200, 400, 600, 800, 1000]),
    "shrink": hp.hp.choice('shrink', [0, 1, 2, 4, 8, 10, 15, 20, 100, 200, 400])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    # Optimize
    best = fmin(fn=objective, space=price_cbf_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    print(best)