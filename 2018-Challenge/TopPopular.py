class TopOfThePops(object):

    def __init__(self, URM):
        self.popularItems = None

        self.URM_csr = URM
    
    def fit(self):

        # Compute most popular items

        item_popularity = (self.URM_csr > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        # Obtain most popular indices
        self.popularItems = np.argsort(item_popularity)

        # Flip (order to higher to lower)
        self.popularItems = np.flip(self.popularItems, axis=0)


    def recommend(self, user_id):
        unseen_items_mask = np.in1d(self.popularItems, self.URM_csr[user_id].indices,
                                        assume_unique=True, invert = True)

        unseen_items = self.popularItems[unseen_items_mask]
        
        return list(unseen_items[0:10])
        #print(recommended_items.shape)


