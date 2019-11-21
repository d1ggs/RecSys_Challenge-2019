import os

import numpy as np
from scipy import sparse
import copy
import pickle

SEED = 12345678


def sample_test_users(URM_csr, sample_size, sample_threshold=10):
    """Sample test user ids of users that have more interactions than a given threshold"""

    # Get a column vector containing for each user the number of interactions
    candidates = URM_csr.sum(1)

    # Get the users with enough interactions
    indices = np.argwhere(candidates > sample_threshold)[:, 0]

    if sample_size > indices.shape[0]:
        print("The sample requested is larger than the candidate users number. Returning whole vector.")
        return indices

    # Sample users randomly
    return np.random.choice(indices, sample_size, replace=False)


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row + 1]] = value


def split_train_test(URM_all: sparse.csr_matrix, split_fraction: float, rewrite=False):
    '''Split the full URM into a train URM and a test dictionary.
    Also save all objects in Pickle serialized format'''

    # Define paths for pickle data
    ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    TEST_DATA_PATH = ROOT_PROJECT_PATH + "/data/pickled/test_data.pickle"
    URM_TRAIN_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_train.pickle"
    URM_TEST_PATH = ROOT_PROJECT_PATH + "/data/pickled/URM_test.pickle"

    if not rewrite:
        # Load serialized Pickle data
        print("Loading pickled data")

        test_data_in = open(TEST_DATA_PATH, "rb")
        test_data = pickle.load(test_data_in)
        test_data_in.close()

        URM_train_in = open(URM_TRAIN_PATH, "rb")
        URM_train = pickle.load(URM_train_in)
        URM_train_in.close()

        URM_test_in = open(URM_TEST_PATH, "rb")
        URM_test = pickle.load(URM_test_in)
        URM_test_in.close()

    else:

        print("Splitting data into train and test...")
        sample_size = int((1 - split_fraction) * URM_all.shape[0])
        indices = sample_test_users(URM_all, sample_size)

        URM_test = copy.deepcopy(URM_all)
        URM_train = copy.deepcopy(URM_all)

        test_data = {}

        # Convert to lil matrix to directly access elements in a faster way than CSR

        URM_train = sparse.lil_matrix(URM_train)

        # Set to 0 ten random elements into train matrix for each designated test user

        for i in range(URM_all.shape[0]):
            if i in indices:
                # It is a test user

                # Get the corresponding row from the URM
                row = URM_all[i, :]

                # Find the nonzero elements (interactions) indices and shuffle them
                interactions = np.argwhere(row > 0)
                np.random.shuffle(interactions)

                # Pick 10 to hold out to compute MAP@10
                test_data[i] = interactions[:, 1][:10]

                # Set to 0 the corresponding interaction in the train matrix
                for inter in interactions[:10]:
                    URM_train[i, inter] = 0
            else:
                # It is not a test user, fill the row with zeros in test matrix
                csr_row_set_nz_to_val(URM_test, i)

        # Eliminate zeroes from the CSR matrix (useless)
        URM_test.eliminate_zeros()

        URM_train = URM_train.tocsr()
        URM_test = URM_test.tocsr()

        # print(URM_train.nnz)
        # print(URM_test.nnz)

        # Serialize the objects with Pickle, for faster loading

        print("Saving Pickle files into /data/pickled")

        test_data_out = open(TEST_DATA_PATH, "wb")
        pickle.dump(test_data, test_data_out)
        test_data_out.close()

        URM_train_out = open(URM_TRAIN_PATH, "wb")
        pickle.dump(URM_train, URM_train_out)
        URM_train_out.close()

        URM_test_out = open(URM_TEST_PATH, "wb")
        pickle.dump(URM_test, URM_test_out)
        URM_test_out.close()

    return URM_train, URM_test, test_data
