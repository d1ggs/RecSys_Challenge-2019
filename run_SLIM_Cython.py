from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from evaluation.Evaluator import Evaluator
from utils.helper import Helper
from utils.run import RunRecommender
if __name__ == '__main__':


    params = {"epochs": 20, "topK": 1}
    recommender = SLIM_BPR_Cython
    RunRecommender.evaluate_on_test_set(recommender,params)
