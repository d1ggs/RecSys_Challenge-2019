import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from PriceCBF import PriceCBF
from utils.run import RunRecommender
import numpy as np


### Step 1 : defining the objective function
def objective(params):
    print('############################################################')
    print(params)
    loss = - RunRecommender.evaluate_on_test_set(PriceCBF, params)
    return loss

price_cbf_space = {
    # "topK": hp.hp.choice('topK', [0, 20, 40, 60, 80, 100, 200, 400, 600]),
    # "shrink": hp.hp.choice('shrink', [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20])
    "topK": hp.hp.choice('topK', np.arange(0, 500, 20)),
    "shrink": hp.hp.uniformint('shrink', 0, 20)
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    # Optimize
    best = fmin(fn=objective, space=price_cbf_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate={'topK': 200, 'shrink':5 })

    ### best will the return the the best hyperparameter set

    print(best)