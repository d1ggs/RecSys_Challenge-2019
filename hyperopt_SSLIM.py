import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from SSLIMElasticNetRecommender import MultiThreadSSLIM_ElasticNet
from utils.run import RunRecommender
from utils.helper import Helper


### Step 1 : defining the objective function
def objective(params):
    print("##################################################")
    print("Current parameters:")
    print(params)
    params["random_state"] = 1234
    params["topK"] = int(params["topK"])
    loss = - RunRecommender.evaluate_on_validation_set(MultiThreadSSLIM_ElasticNet, params, Kfold=10, user_group="warm")
    return loss

elasticnet_space = {
    "alpha": hp.hp.uniform('alpha', 0.001, 0.01),
    "l1_ratio": hp.hp.uniform('l1_ratio', 1e-5, 1e-3),
    "positive_only": hp.hp.choice('positive_only', [True, False]),
    "topK": hp.hp.quniform('topK', 0, 100, 5),
    "side_alpha": hp.hp.uniform('side_alpha', 0, 100),
    "bm_25_all": hp.hp.choice('bm_25_all', [True, False]),
    "bm_25_urm": hp.hp.choice('bm_25_urm', [True, False]),
    "bm_25_icm": hp.hp.choice('bm_25_icm', [True, False])

    #"fit_intercept": hp.hp.choice('fit_intercept', [True, False]),
    #"max_iter": hp.hp.choice('max_iter', np.arange(100, 1000, 100, dtype=int)),
    #"tol": hp.hp.choice('tol', [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]),
    #"selection": hp.hp.choice('selection', ['random', 'cyclic']),
}


if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 50

    previous_best = {"alpha": 1.0, "l1_ratio": 0.001, "positive_only": False, "fit_intercept": False, "max_iter": 100,
                     "tol": 1e-4, "selection": "random", "random_state": 1234}

    old_best = {'alpha': 0.003890771067122292, 'l1_ratio': 2.2767573538452768e-05, 'positive_only': True, 'topK': 100, "random_state": 1234}
    new_best = {'alpha': 0.0023512567548654, 'l1_ratio': 0.0004093694334328875, 'positive_only': True, 'topK': 25, "random_state": 1234}

    sslim = {'alpha': 0.0024081648139725204, 'l1_ratio': 0.0007553368138338653, 'positive_only': False,
             'side_alpha': 3.86358712510434, 'topK': 65, "bm_25_urm": False, "bm_25_icm": False, "bm_25_all": False}

    # Optimize
    best = fmin(fn=objective, space=elasticnet_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True,
                points_to_evaluate=[old_best, new_best, sslim])

    ### best will the return the the best hyperparameter set

    params = space_eval(elasticnet_space, best)
    print("############### Best parameters ###############")
    params["topK"] = int(params["topK"])
    print(params)

    print("############### Test set performance ###############")
    RunRecommender.evaluate_on_test_set(MultiThreadSSLIM_ElasticNet, params, Kfold=6, user_group="warm")
