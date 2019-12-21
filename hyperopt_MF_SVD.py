import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()

### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_test_set(PureSVDRecommender, params)
    return loss

MF_space = {
    "num_factors": hp.hp.choice('num_factors', np.arange(1, 1000)),
    "random_seed": hp.hp.choice('random_seed', [1234])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50


    # Optimize
    best = fmin(fn=objective, space=MF_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True,)

    ### best will the return the the best hyperparameter set

    print(best)
