import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from ALS import AlternatingLeastSquare
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper
from utils.schleep import computer_sleep

helper = Helper()
import gc


N_K_FOLD = 10

### Step 1 : defining the objective function
def objective(params):
    print('###########################################')
    params["n_factors"] = int(params["n_factors"])
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(AlternatingLeastSquare, params, Kfold=N_K_FOLD, parallel_fit=False, parallelize_evaluation=True, user_group="warm")
    return loss


search_space = {
    "n_factors": hp.hp.quniform('n_factors', 200, 1000, 30),

    "regularization": hp.hp.uniform('regularization', 0.0001, 0.21),
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    opt_eval = {'n_factors': 680, 'regularization': 0.08905996811679368}

    # Optimize
    best = fmin(fn=objective, space=search_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=opt_eval)

    params = space_eval(search_space, best)

    ### best will the return the the best hyperparameter set

    print("Best parameters:")
    print(params)
    params["topK"] = int(params["topK"])

    ### best will the return the the best hyperparameter set

    params = space_eval(search_space, best)
    print("############### Best parameters ###############")
    print(params)

    print("############### Test set performance ###############")
    RunRecommender.evaluate_on_test_set(AlternatingLeastSquare, params, parallel_fit=False, Kfold=N_K_FOLD,
                                        parallelize_evaluation=True, user_group="warm")

    computer_sleep(verbose=False)

