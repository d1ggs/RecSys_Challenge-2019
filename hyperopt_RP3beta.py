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
    "alpha": hp.hp.uniform('alpha', 1.5, 2.5),
    "beta": hp.hp.uniform('beta', 0.3, 0.6),
    "min_rating": 0,
    "topK": hp.hp.choice('topK', np.arange(580, 700, 2)),
    "implicit": False,
    "normalize_similarity": False
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    opt = {'alpha': 1.0043111988071665, 'beta': 0.0029837192341647823, 'implicit': False, 'min_rating': 0, 'normalize_similarity': False, 'topK': 680}
    opt_minor = {'alpha': 1.009077614740585, 'beta': 0.029849929646199647, 'implicit': False, 'min_rating': 0, 'normalize_similarity': False, 'topK': 600}


    RunRecommender.evaluate_on_validation_set(RP3betaRecommender, opt)
    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[opt_minor, opt])

    ### best will the return the the best hyperparameter set

    print(best)
    RunRecommender.evaluate_on_test_set(RP3betaRecommender, best)