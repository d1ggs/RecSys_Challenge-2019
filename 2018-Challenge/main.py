# from abc import ABC, abstractmethod

from data_utilities import build_URM, write_submission, get_targets, build_ICM
from ContentBasedFilter import ContentBasedFilter
    
if __name__ == "__main__":

    target_playlists = get_targets("data/target_playlists.csv")
    URM_data = build_URM("data/train.csv")
    ICM_album = build_ICM("data/tracks.csv", "album_id")
    ICM_artist = build_ICM("data/tracks.csv", "artist_id")
    #topPop = TopOfThePops(URM_data)

    #topPop.fit()

    ICM_data = [ICM_artist, ICM_album]

    CF = ContentBasedFilter(URM_data, ICM_data)

    CF.fit()

    write_submission(CF, target_playlists)

    #print(URM_csr.toarray())


