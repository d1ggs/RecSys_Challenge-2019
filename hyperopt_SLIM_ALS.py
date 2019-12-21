import hyperopt as hp
from hyperopt import Trials, fmin, STATUS_OK
from Hybrid_ALS_SLIMElastic import HybridALSElasticNet
from utils.run import RunRecommender
from utils.helper import Helper



hybrid = HybridALSElasticNet(Helper().URM_train_validation, mode="validation")


### Step 1 : defining the objective function
def objective(params):
    print("Current parameters:")
    print(params)
    loss = - RunRecommender.evaluate_hybrid_weights_validation(hybrid, params)
    return loss

als_space = {
    "SLIM_weight": hp.hp.uniform('SLIM_weight', 0.7, 1.0),
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True)

    ### best will the return the the best hyperparameter set

    print(best)