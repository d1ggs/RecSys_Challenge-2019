import hyperopt as hp
from hyperopt import Trials, fmin, space_eval
from utils.run import RunRecommender
from Hybrid_User_CBF_Regional_TopPop import HybridUserCBFRegionalTopPop
import numpy as np


### Step 1 : defining the objective function
def objective(params):
    print('###########################################')
    print(params)
    params['similarity'] = "cosine"
    loss = - RunRecommender.evaluate_on_validation_set(HybridUserCBFRegionalTopPop, params, Kfold=10, user_group="cold", parallel_fit=False, parallelize_evaluation=True)
    return loss

user_cbf_space = {
    "topK": hp.hp.choice('topK', np.arange(0, 800, 5)),
    "shrink": hp.hp.uniformint('shrink', 0, 50),
    #"similarity": hp.hp.choice('similarity', ["cosine"]),
    "suppress_interactions": hp.hp.choice('suppress_interactions', [True, False]),
    "bm_25_norm": hp.hp.choice('bm_25_norm', [True, False]),
    "normalize": hp.hp.choice('normalize', [True, False]),
    "top_pop_weight": hp.hp.uniform('top_pop_weight', 0, 1)
}

if __name__ == '__main__':
    ### step 3 : storing the results of every iteration
    bayes_trials = Trials()
    MAX_EVALS = 100

    previous = {'normalize': True, 'shrink': 5, 'similarity': "cosine", 'suppress_interactions': False, 'topK': 200}
    last = {'normalize': True, 'shrink': 1.0, 'similarity': "dice", 'suppress_interactions': True, 'topK': 93*5}
    old_best = {"top_pop_weight": 0.02139131367609725, "topK": 765, "shrink": 6, "normalize": True, "similarity": "jaccard",
            "suppress_interactions": False}

    # Optimize
    best = fmin(fn=objective, space=user_cbf_space, algo=hp.tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, verbose=True, points_to_evaluate=[previous, last])

    ### best will the return the the best hyperparameter set

    print(best)
    params = space_eval(user_cbf_space, best)
    print(params)

    print("############### Performance on test set #################")
    MAP = RunRecommender.evaluate_on_test_set(HybridUserCBFRegionalTopPop, params, Kfold=10, user_group="cold", parallel_fit=False, parallelize_evaluation=True)