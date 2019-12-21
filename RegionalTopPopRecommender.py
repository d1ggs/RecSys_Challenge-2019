from TopPopularRecommender import TopPopRecommender
import numpy as np


class RegionalTopPopRecommender(object):


    def __init__(self, URM, UCM_list):
        self.URM = URM
        self.UCM_list = UCM_list

        # SCORE WEIGHTS - COLD USERS
        self.toppop = TopPopRecommender(self.URM)

        self.regions = list(np.unique(UCM_list[0].indices))


    def fit(self):

        self.toppop.fit()

        self.regions_top_scores = []
        for region in self.regions:
            region_mask = self.UCM_list[0].indices == region
            top_pop = TopPopRecommender(self.URM[region_mask])
            top_pop.fit()
            scores = top_pop.get_weighted_popularity_score()
            self.regions_top_scores.append(scores)



    def recommend(self, user_id, at=10):

        self.final_scores = self.scores(user_id)

        ranking = self.final_scores.argsort()[::-1]

        return ranking[0:at]


    def scores(self, user_id):
        # Regional topPop
        start_region = self.UCM_list[0].indptr[user_id]
        end_region = self.UCM_list[0].indptr[user_id + 1]

        if end_region - start_region > 0:
            user_region = self.UCM_list[0].indices[start_region]
            scores = self.regions_top_scores[self.regions.index(user_region)]
        else:
            # If user has no region, use general topPop
            scores = self.toppop.get_weighted_popularity_score()
        return scores
