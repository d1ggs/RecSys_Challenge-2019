from TopPopularRecommender import TopPopRecommender
import numpy as np

from utils.helper import Helper


class RegionalTopPopRecommender(object):


    def __init__(self, URM):
        self.URM = URM
        self.UCM_region = Helper().load_ucm_region()

        # SCORE WEIGHTS - COLD USERS
        self.toppop = TopPopRecommender(self.URM)

        self.regions = list(np.unique(self.UCM_region.indices))


    def fit(self):

        self.toppop.fit()

        self.regions_top_scores = []
        for region in self.regions:
            region_mask = self.UCM_region.indices == region
            top_pop = TopPopRecommender(self.URM[region_mask])
            top_pop.fit()
            scores = top_pop.item_popularity
            self.regions_top_scores.append(scores)



    def recommend(self, user_id, at=10, exclude_seen=False):

        #TODO implement exclude seen for use on warm users

        self.final_scores = self.scores(user_id)

        ranking = self.final_scores.argsort()[::-1]

        return ranking[0:at]


    def scores(self, user_id):
        # Regional topPop
        start_region = self.UCM_region.indptr[user_id]
        end_region = self.UCM_region.indptr[user_id + 1]

        if end_region - start_region > 0:
            user_region = self.UCM_region.indices[start_region]
            scores = self.regions_top_scores[self.regions.index(user_region)]
        else:
            # If user has no region, use general topPop
            scores = self.toppop.item_popularity
        return scores
