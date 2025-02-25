import hyperopt as hp
from hyperopt import Trials, fmin, space_eval, STATUS_OK
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from evaluation.Evaluator import Evaluator
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper


helper = Helper()

N_KFOLD = 10

MAX_EVALS = 100

evaluator = Evaluator()

### Step 1 : defining the objective function
def objective(params):
    params["topK"] = int(params["topK"])
    params["batch_size"] = int(params["batch_size"])
    params["random_seed"] = 1234
    params["epochs"] = 30
    print("############ Current parameters ############")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(SLIM_BPR_Cython, params, Kfold=N_KFOLD, parallelize_evaluation=False, user_group="warm")
    return loss

slim_bpr_space = {
    "batch_size": hp.hp.quniform('batch_size', 0, 1000, 50),
    "lambda_i": hp.hp.uniform('lambda_i', 0, 0.1),
    "lambda_j": hp.hp.uniform('lambda_j', 0, 0.1),
    "learning_rate": hp.hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-1)),
    "topK": hp.hp.quniform('topK', 0, 1000, 50),
    "gamma": hp.hp.uniform('gamma', 0.5, 1.0),
    "beta_1": hp.hp.uniform('beta_1', 0.5, 1.0),
    "beta_2": hp.hp.uniform('beta_2', 0.5, 1.0)
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()

    # Optimize
    best = fmin(fn=objective, space=slim_bpr_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    params = space_eval(slim_bpr_space, best)

    params["topK"] = int(params["topK"])
    params["batch_size"] = int(params["batch_size"])
    params["random_seed"] = 1234
    params["epochs"] = 30

    print("############### Best parameters ###############")

    print(params)
    RunRecommender.evaluate_on_test_set(SLIM_BPR_Cython, params, Kfold=N_KFOLD, parallelize_evaluation=False, user_group="warm")

