from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from utils.helper import Helper
from utils.run import RunRecommender
import matplotlib.pyplot as plt
from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch

pset = [{"lr": 0.1, "epochs": 1000, "nf": 5},
        {"lr": 0.1, "epochs": 20, "nf": 10},
        {"lr": 0.1, "epochs": 20, "nf": 50},
        {"lr": 0.1, "epochs": 20, "nf": 100},
        {"lr": 0.1, "epochs": 20, "nf": 200},
        {"lr": 0.5, "epochs": 20, "nf": 300},
        {"lr": 0.1, "epochs": 20, "nf": 400}]

helper = Helper()

# Train and test data are now loaded by the helper
#URM_train, test_data = helper.get_train_evaluation_data(resplit=False, split_fraction=0, leave_out=1)
num_users = helper.get_number_of_users()

map = []
i = 0

for params in pset:
    i += 1
    print("Parameter set", i)

    recommender = MF_MSE_PyTorch(helper.URM_train_validation)
    recommender.fit(num_factors=params["nf"])
    #map.append(RunRecommender.perform_evaluation_slim(recommender, test_data, num_users))

plt.plot(map)
plt.ylabel('MAP-10 score')
plt.xlabel('Parameter set')
plt.title("Factors number")
plt.show()
