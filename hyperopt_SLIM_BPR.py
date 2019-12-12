import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper


helper = Helper()


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(SLIM_BPR_Cython, params)
    return loss

slim_cbr_space = {
    "epoch": hp.hp.choice('epoch', [30]),
    "positive_threshold_BPR": hp.hp.choice('positive_threshold_BPR', [None]),
    "train_with_sparse_weights": hp.hp.choice('train_with_sparse_weights', [None]),
    "symmetric": hp.hp.choice('symmetric', [True]),
    "random_seed": hp.hp.choice('random_seed', [1234]),
    "batch_size": hp.hp.choice('batch_size', np.arange(0, 1000, 50)),
    "lambda_i": hp.hp.uniform('lambda_i', 0.00, 0.1),
    "lambda_j": hp.hp.uniform('lambda_j', 0.000, 0.1),
    "learning_rate": hp.hp.uniform('learning_rate', 1e-5, 1e-1),
    "topK": hp.hp.uniform('topK', np.arange(0, 1000, 50)),
    "sgd_mode": hp.hp.choice('sgd_mode', ['adagrad']),
    "gamma": hp.hp.uniform('gamma', 0.5, 1.0),
    "beta_1": hp.hp.uniform('beta_1', 0.5, 1.0),
    "beta_2": hp.hp.uniform('beta_2', 0.5, 1.0)
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50




    # Optimize
    best = fmin(fn=objective, space=slim_cbr_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    print(best)