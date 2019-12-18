import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from ItemBasedCBF import ItemBasedCBF
from utils.run import RunRecommender
from utils.helper import Helper
import numpy as np

hybrid = ItemBasedCBF(Helper().URM_train_validation, mode="validation")
### Step 1 : defining the objective function
def objective(params):
    loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss

cbf_space = {
    "weight_asset": hp.hp.uniform('weight_asset', 0.0, 1.0),
    "weight_price": hp.hp.uniform('weight_price', 0.0, 1.0),
    "weight_sub_class": hp.hp.uniform('weight_sub_class', 0.0, 1.0)
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100


    # Optimize
    best = fmin(fn=objective, space=cbf_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    print(best)