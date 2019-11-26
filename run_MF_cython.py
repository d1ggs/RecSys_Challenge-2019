from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from utils.helper import Helper
from utils.run import RunRecommender
import matplotlib.pyplot as plt

pset = [{"lr": 0.1, "epochs": 20, "nf": 5},
        {"lr": 0.1, "epochs": 20, "nf": 10},
        {"lr": 0.1, "epochs": 20, "nf": 50},
        {"lr": 0.1, "epochs": 20, "nf": 100},
        {"lr": 0.1, "epochs": 20, "nf": 200},
        {"lr": 0.5, "epochs": 20, "nf": 300},
        {"lr": 0.1, "epochs": 20, "nf": 400}]

helper = Helper()

# Train and test data are now loaded by the helper
URM_train, test_data = helper.get_train_test_data(resplit=False, split_fraction=0, leave_out=1)
print(helper.get_number_of_users())
num_users = helper.get_number_of_users()
exit(0)

map = []
i = 0

for params in pset:
    i += 1
    print("Parameter set", i)
    recommender = MatrixFactorization_FunkSVD_Cython(URM_train, recompile_cython=False, verbose=False)
    recommender.fit(learning_rate=params["lr"], epochs=params["epochs"], num_factors=params["nf"], random_seed=1234)
    map.append(RunRecommender.perform_evaluation_slim(recommender, test_data, num_users))

plt.plot(map)
plt.ylabel('MAP-10 score')
plt.xlabel('Parameter set')
plt.title("Factors number")
plt.show()
