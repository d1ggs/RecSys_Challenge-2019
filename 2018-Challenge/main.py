# from abc import ABC, abstractmethod

from data_utilities import build_URM, write_submission, get_targets

from Notebooks_utils.data_splitter import train_test_holdout

# from Models.SLIMRecommender import SLIMRecommender

from Notebooks_utils.evaluation_function import evaluate_algorithm

from Models.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet


if __name__ == "__main__":

    target_playlists = get_targets("data/target_playlists.csv")
    URM_all = build_URM("data/train.csv")

    URM_test, URM_train = train_test_holdout(URM_all, train_perc=0.8)
    # ICM_album = build_ICM("data/tracks.csv", "album_id")
    # ICM_artist = build_ICM("data/tracks.csv", "artist_id")
    
    # topPop = TopOfThePops(URM_data)
    # topPop.fit()

    # ICM_data = [ICM_artist, ICM_album]

    # CF = CollaborativeFilter(URM_train)

    # CF.fit()

    #print("Fitting model...")
    #recommender = SLIMRecommender(URM_train)
    #recommender.fit(epochs=1000)
    #print("Done!")

    recommender = MultiThreadSLIM_ElasticNet(URM_train)
    print("Fitting model...")
    recommender.fit()
    print("Done!")

    evaluate_algorithm(URM_test, recommender)

    write_submission(recommender, target_playlists)

    #print(URM_csr.toarray())


