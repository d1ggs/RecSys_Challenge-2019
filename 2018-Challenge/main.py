from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
import numpy as np
from data_utilities import prepare_data, write_submission, get_targets

class RecSys(ABC):
    def __init__(self, *args, **kwargs):
        self.URM_csr = None
        super().__init__(*args, **kwargs)

    @abstractmethod
    def fit(self, URM_tuples):
        pass

    @abstractmethod
    def predict(self, user_id):
        pass

class TopOfThePops(RecSys):

    def __init__(self, *args, **kwargs):
        self.popularItems = None
        super().__init__(*args, **kwargs)

    def fit(self, URM_tuples):
        ones = np.array([1]*len(URM_data)) # Assign 1 where there has been an interaction
    
        # Build COO representation
        rows, cols = [], []

        for URM_tuple in URM_tuples:
            rows.append(int(URM_tuple[0]))
            cols.append(int(URM_tuple[1]))

        #print(len(rows))
        #print(cols)

        # Obtain CSR representation
        
        self.URM_csr = csr_matrix( (ones, (np.array(rows), np.array(cols)) ) )

        # Compute most popular items

        item_popularity = (self.URM_csr > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        # Obtain most popular indices
        self.popularItems = np.argsort(item_popularity)

        # Flip (order to higher to lower)
        self.popularItems = np.flip(self.popularItems, axis=0)


    def predict(self, user_id):
        unseen_items_mask = np.in1d(self.popularItems, self.URM_csr[user_id].indices,
                                        assume_unique=True, invert = True)

        unseen_items = self.popularItems[unseen_items_mask]
        
        return list(unseen_items[0:10])
        #print(recommended_items.shape)


class GlobalEffects(RecSys):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, URM_tuples):
        pass

    def predict(self, user_id):
        pass

if __name__ == "__main__":

    target_playlists = get_targets("data/target_playlists")
    URM_data = prepare_data("data/train.csv")
    
    topPop = TopOfThePops()

    topPop.fit(URM_data)

    write_submission(topPop, target_playlists[1:])

    #print(URM_csr.toarray())


