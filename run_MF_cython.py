from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from utils.helper import Helper
from utils.run import RunRecommender
import matplotlib as plt

pset = [{"lr": 0.1, "epochs": 300, "nf": 10},
        {"lr": 0.05, "epochs": 300, "nf": 10},
        {"lr": 0.01, "epochs": 300, "nf": 10},
        {"lr": 0.005, "epochs": 300, "nf": 10},
        {"lr": 0.001, "epochs": 300, "nf": 10},
        {"lr": 0.0005, "epochs": 300, "nf": 10},
        {"lr": 0.0001, "epochs": 300, "nf": 10}]

helper = Helper()

# Train and test data are now loaded by the helper
URM_train, test_data = helper.get_train_test_data()

map = []

for params in pset:
    recommender = MatrixFactorization_BPR_Cython(URM_train, recompile_cython=False, verbose=False)
    recommender.fit(learning_rate=params["lr"], epochs=params["epochs"], num_factors=params["nf"])
    map.append(RunRecommender.perform_evaluation_slim(recommender, test_data))

plt.plot(map)
plt.ylabel('MAP-10 score')
plt.xlabel('Parameter set')
plt.show()
