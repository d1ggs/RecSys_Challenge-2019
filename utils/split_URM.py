import os

import numpy as np
from scipy import sparse
import copy
import pickle

SEED = 12345678


def sample_test_users(URM_csr, sample_size, sample_threshold=0):
    """Sample test user ids of users that have more interactions than a given threshold"""

    # Get a column vector containing for each user the number of interactions
    candidates = URM_csr.sum(1)

    # Get the users with enough interactions
    indices = np.argwhere(candidates >= sample_threshold)[:, 0]

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


def split_train_test(URM_all: sparse.csr_matrix, split_fraction: float, leave_out=1):
    '''Split the full URM into a train URM and a test dictionary.
    Also save all objects in Pickle serialized format'''

    print("Splitting data into train and test...")
    sample_size = int((1 - split_fraction) * URM_all.shape[0])

    URM_train_test, test_data = hold_out(URM_all, leave_out, sample_size)
    URM_train_validation, validation_data = hold_out(URM_train_test, leave_out, sample_size)

    return URM_train_validation, URM_train_test, validation_data, test_data


def hold_out(URM, leave_out, sample_size):
    """Set to 0 ten random elements into train matrix for each designated test user"""

    URM_hold_out = URM.copy()
    indices = sample_test_users(URM, sample_size, sample_threshold=leave_out)
    hold_out_data = {}

    # Convert to lil matrix to directly access elements in a faster way than CSR
    URM_hold_out = sparse.lil_matrix(URM_hold_out)


    for i in range(URM.shape[0]):
        if i in indices:
            # It is a test user

            # Get the corresponding row from the URM
            row = URM[i, :]

            # Find the nonzero elements (interactions) indices and shuffle them
            interactions = np.argwhere(row > 0)
            np.random.shuffle(interactions)

            # Pick 1 to hold out to compute MAP@10
            hold_out_data[i] = np.array(interactions[:, 1][:leave_out])

            # Set to 0 the corresponding interaction in the train matrix
            for inter in interactions[:leave_out]:
                URM_hold_out[i, inter] = 0

    URM_hold_out = URM_hold_out.tocsr()

    return URM_hold_out, hold_out_data
