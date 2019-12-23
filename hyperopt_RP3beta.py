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
    "alpha": hp.hp.uniform('alpha', 0.2, 0.4),
    "beta": hp.hp.uniform('beta', 0.08, 0.3),
    "min_rating": 0,
    "topK": hp.hp.choice('topK', np.arange(50, 150, 2)),
    # "shrink": hp.hp.choice('shrink', np.arange(0, 100, 5)),
    "implicit": True,
    "normalize_similarity": True
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    opt = {'alpha': 0.31932803725825626, 'beta': 0.19051435359666555, 'implicit': True, 'min_rating': 0, 'normalize_similarity': True, 'topK': 56}





    # RunRecommender.evaluate_on_validation_set(RP3betaRecommender, opt)
    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt)

    ### best will the return the the best hyperparameter set

    print(best)
    RunRecommender.evaluate_on_test_set(RP3betaRecommender, best)

    """ 
    MAP to beat: .03581871259565834 val
    .041 test
    """