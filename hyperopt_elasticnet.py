import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper

helper = Helper()
import gc


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_test_set(MultiThreadSLIM_ElasticNet, params)
    return loss

elasticnet_space = {
    "alpha": hp.hp.uniform('alpha', 0.0, 0.1),
    "l1_ratio": hp.hp.uniform('l1_ratio', 1e-6, 1e-3),
    "positive_only": hp.hp.choice('positive_only', [True, False]),
    "topK": hp.hp.choice('topK', np.arange(0, 1000, 10)),
    #"fit_intercept": hp.hp.choice('fit_intercept', [True, False]),
    #"max_iter": hp.hp.choice('max_iter', np.arange(100, 1000, 100, dtype=int)),
    #"tol": hp.hp.choice('tol', [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]),
    #"selection": hp.hp.choice('selection', ['random', 'cyclic']),
    "random_state": hp.hp.choice('random_state', [1234])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    previous_best = {"alpha": 1.0, "l1_ratio": 0.001, "positive_only": False, "fit_intercept": False, "max_iter": 100,
                     "tol": 1e-4, "selection": "random", "random_state": 1234}

    new_best = {'alpha': 0.003890771067122292, 'l1_ratio': 2.2767573538452768e-05, 'positive_only': True, 'topK': 100}

    # Optimize
    best = fmin(fn=objective, space=elasticnet_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True,
                points_to_evaluate=[previous_best, new_best])

    ### best will the return the the best hyperparameter set

    print(best)