import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from RP3betaRecommender import RP3betaRecommender
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()
import gc


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(RP3betaRecommender, params)
    return loss

als_space = {
    "alpha": hp.hp.uniform('alpha', 1.0, 4.0),
    "beta": hp.hp.uniform('beta', 0.0, 1.0),
    "min_rating": 0,
    "topK": hp.hp.choice('topK', np.arange(0, 800, 40)),
    "implicit": False,
    "normalize_similarity": False
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, )

    ### best will the return the the best hyperparameter set

    print(best)