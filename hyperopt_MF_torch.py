import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from utils.run import RunRecommender
import numpy as np
from utils.helper import Helper
from evaluation.Evaluator import Evaluator

helper = Helper()

### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_on_test_set(MF_MSE_PyTorch, params)
    return loss

MF_space = {
    "num_factors": hp.hp.uniformint('num_factors', 1, 5000),
    "batch_size": hp.hp.choice('batch_size', [32, 64, 128, 256, 512]),
    "learning_rate": hp.hp.loguniform('learning_rate', 1e-6, 1e-2),
    "epochs": hp.hp.uniformint('epochs', 1, 1000),
    "validation_every_n": hp.hp.choice("validation_every_n",[10]),
    "validation_metric": hp.hp.choice('validation_metric', ['MAP']),
    "lower_validations_allowed": hp.hp.choice('lower_validations_allowed', [2]),
    "evaluator_object": hp.hp.choice('evaluator_object', [Evaluator(test_mode=False)])
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 20


    # Optimize
    best = fmin(fn=objective, space=MF_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True,)

    ### best will the return the the best hyperparameter set

    print(best)
