from Hybrid import HybridUCFICFRecommender
from evaluation.Evaluator import Evaluator
from utils.helper import Helper
from utils.split_URM import split_train_test
from HybridCBCBF import HybridCBCBFRecommender

if __name__ == '__main__':

    helper = Helper(resplit=False)

    URM_train = helper.URM_train_validation

    evaluator_validation = Evaluator()
    evaluator_test = Evaluator(test_mode=True)

    from Legacy.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    recommender_class = HybridCBCBFRecommender

    parameterSearch = SearchBayesianSkopt(recommender_class,
                                          evaluator_validation=evaluator_validation,
                                          evaluator_test=None)

    from Legacy.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    from skopt.space import Real, Integer, Categorical

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["cf_weight"] = Integer(0, 100)
    hyperparameters_range_dictionary["cbf_weight"] = Integer(0, 100)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],\
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    output_folder_path = "result_experiments/"

    import os

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 40
    metric_to_optimize = "MAP"

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=5,
                           save_model="no",
                           output_folder_path=output_folder_path,
                           output_file_name_root=recommender_class.RECOMMENDER_NAME,
                           metric_to_optimize=metric_to_optimize
                           )

    from Legacy.Base.DataIO import DataIO

    data_loader = DataIO(folder_path=output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

    best_parameters = search_metadata["hyperparameters_best"]
    print(best_parameters)