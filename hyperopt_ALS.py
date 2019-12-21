import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from ALS import AlternatingLeastSquare
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()
import gc


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(AlternatingLeastSquare, params)
    return loss

als_space = {
    "n_factors": hp.hp.choice('n_factors', np.arange(200, 1000, 30)),

    "regularization": hp.hp.uniform('regularization', 0.0001, 0.21),
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 40

    opt_eval = {'n_factors': 680, 'regularization': 0.08905996811679368}

    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt_eval)

    ### best will the return the the best hyperparameter set

    print(best)