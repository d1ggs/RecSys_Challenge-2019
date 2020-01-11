import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from RP3betaRecommender import RP3betaRecommender
from utils.schleep import computer_sleep
from utils.run import RunRecommender
from utils.helper import Helper

helper = Helper()

N_KFOLD = 10
MAX_EVALS = 100

### Step 1 : defining the objective function
def objective(params):
    params["topK"] = int(params["topK"])
    print("############ Current parameters ############")
    print(params)
    loss = - RunRecommender.evaluate_on_validation_set(RP3betaRecommender, params, Kfold=N_KFOLD, parallelize_evaluation=True, user_group="warm")
    return loss

als_space = {
    "alpha": hp.hp.uniform('alpha', 0.2, 0.4),
    "beta": hp.hp.uniform('beta', 0.08, 0.3),
    "topK": hp.hp.quniform('topK', 50, 150, 2),
    # "shrink": hp.hp.choice('shrink', np.arange(0, 100, 5)),
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()

    opt = {'alpha': 0.31932803725825626, 'beta': 0.19051435359666555, 'implicit': True, 'min_rating': 0, 'normalize_similarity': True, 'topK': 56}
    new_opt ={'alpha': 1.009077614740585, 'beta': 0.029849929646199647, 'implicit': False, 'min_rating': 0, 'normalize_similarity': False, 'topK': 600}


    # RunRecommender.evaluate_on_validation_set(RP3betaRecommender, opt)
    # Optimize
    best = fmin(fn=objective, space=als_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[opt, new_opt])

    ### best will the return the the best hyperparameter set

    best = space_eval(als_space, best)

    print("############### Best parameters ###############")


    print(best)
    RunRecommender.evaluate_on_test_set(RP3betaRecommender, best, Kfold=N_KFOLD, parallelize_evaluation=True, user_group="warm")

    computer_sleep(verbose=False)

    """ 
    MAP to beat: .03581871259565834 val
    .041 test
    """