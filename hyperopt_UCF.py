import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from UserCollaborativeFilter import UserCollaborativeFilter
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()
import gc


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(UserCollaborativeFilter, params)
    return loss

als_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 500, 10)),

    "shrink": hp.hp.choice('shrink', np.arange(0, 60, 1))
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    user_cf_parameters = {"topK": 410,
                          "shrink": 0}
    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=user_cf_parameters)

    ### best will the return the the best hyperparameter set

    print(best)