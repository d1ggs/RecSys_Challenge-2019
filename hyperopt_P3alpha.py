import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from P3alphaRecommender import P3alphaRecommender
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()
import gc


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(P3alphaRecommender, params)
    return loss

als_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 800, 40)),

    "alpha": hp.hp.uniform('alpha', 1.0, 4.0),
    "min_rating": 0,
    "implicit": hp.hp.choice('implicit', [False, True]),
    "normalize_similarity": False
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 150

    {'alpha': 1.0022745975302572, 'implicit': 1, 'topK': 18}
    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt)

    ### best will the return the the best hyperparameter set

    print(best)