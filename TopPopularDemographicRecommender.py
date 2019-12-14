import numpy as np
from utils.run import RunRecommender
from utils.helper import Helper


class TopPopDemographicRecommender:

    def __init__(self, URM_CSR, mode="validation"):
        self.URM_CSR = URM_CSR
        self.item_popularity = None
        self.user_data_dict, self.cluster_dict, self.region_dict, self.age_dict = \
            Helper().get_demographic_info(verbose=False, mode=mode)
        self.cluster_item_popularity = {}
        self.age_item_popularity = {}
        self.region_item_popularity = {}
        self.cluster_popular_items = {}
        self.age_popular_items = {}
        self.region_popular_items = {}


    def fit(self):
        # --- CSR Matrix Computation ---
        #Training data loading and its conversion to a CSR matrix is done in run.py with the functions provided by Helper

        # --- Actual Popularity computation ---
        # Calculate item popularity by summing for each item the rating of every use

        for cluster in self.cluster_dict.keys():
            user_list = self.cluster_dict[cluster]
            mask = np.isin(np.arange(0, self.URM_CSR.shape[0]), np.array(user_list))
            filtered_URM = self.URM_CSR[mask]
            cluster_item_popularity = (filtered_URM > 0).sum(axis=0)
            self.cluster_item_popularity[cluster] = np.array(cluster_item_popularity).squeeze()
            popular_items_cluster = np.argsort(cluster_item_popularity)
            self.cluster_popular_items[cluster] = np.flip(popular_items_cluster, axis=0)

        for age in self.age_dict.keys():
            user_list = self.age_dict[age]
            mask = np.isin(np.arange(0, self.URM_CSR.shape[0]), np.array(user_list))
            filtered_URM = self.URM_CSR[mask]
            age_item_popularity = (filtered_URM > 0).sum(axis=0)
            self.age_item_popularity[age] = np.array(age_item_popularity).squeeze()
            popular_items_age = np.argsort(age_item_popularity)
            self.age_popular_items[age] = np.flip(popular_items_age, axis=0)

        for region in self.region_dict.keys():
            user_list = self.region_dict[region]
            mask = np.isin(np.arange(0, self.URM_CSR.shape[0]), np.array(user_list))
            filtered_URM = self.URM_CSR[mask]
            region_item_popularity = (filtered_URM > 0).sum(axis=0)
            self.region_item_popularity[region] = np.array(region_item_popularity).squeeze()
            popular_items_region = np.argsort(region_item_popularity)
            self.region_popular_items[region] = np.flip(popular_items_region, axis=0)

        # The most popular playlist is the one with more songs in it
        item_popularity = (self.URM_CSR > 0).sum(axis=0)
        # Squeeze term removes single-dimensional entries from the shape of an array.
        self.item_popularity = np.array(item_popularity).squeeze()

        # Ordering the items according to the popularity values
        self.popular_items = np.argsort(self.item_popularity)

        # Flip order high to lower
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, at=10, exclude_seen=True):
        if user_id in self.user_data_dict:

            obj = self.user_data_dict[user_id]

            if obj.cluster is not None and obj.age is None and obj.region is None:
                recommended_items = self.cluster_popular_items[0:at]

            elif obj.region is None and obj.age is not None and obj.region is None:
                recommended_items = self.age_popular_items[0:at]

            elif obj.cluster is None and obj.age is None and obj.region is not None:
                recommended_items = self.region_popular_items[0:at]

            else:
                recommended_items = None
                print("User with malformed DoubleData object")
                exit()


        elif exclude_seen:
            unseen_items_mask = np.in1d(self.popular_items, self.URM_CSR[user_id].indices, assume_unique=True, invert=True)

            unseen_items = self.popular_items[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popular_items[0:at]

        return recommended_items


if __name__ == "__main__":

    # evaluator = Evaluator()
    # evaluator.split_data_randomly_2()

    top_popular = TopPopDemographicRecommender

    map10 = RunRecommender.evaluate_on_test_set(top_popular, {})

    #print('{0:.128f}'.format(map10))
    print(map10)
