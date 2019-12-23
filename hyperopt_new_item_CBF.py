import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from AssetCBF import AssetCBF
from utils.run import RunRecommender
from ItemBasedCBF import ItemCBF
import numpy as np


### Step 1 : defining the objective function
def objective(params):
    print('###########################################')
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(ItemCBF, params)
    return loss

price_cbf_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 500, 10)),
    "shrink": hp.hp.choice('shrink', np.arange(0, 100, 2))}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100
    opt = {'topK': 200, 'shrink': 5}
    RunRecommender.evaluate_on_validation_set(ItemCBF, opt)
    # Optimize
    best = fmin(fn=objective, space=price_cbf_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    print(best)