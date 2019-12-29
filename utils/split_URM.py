import os
import random

import numpy as np
from scipy import sparse as sps
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
    if not isinstance(csr, sps.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row + 1]] = value


def split_train_test(URM_all: sps.csr_matrix, split_fraction: float, leave_out=1):
    '''Split the full URM into a train URM and a test dictionary.
    Also save all objects in Pickle serialized format'''

    print("Splitting data into train and test...")
    sample_size = int((1 - split_fraction) * URM_all.shape[0])

    URM_train_test, test_data = leave_one_out(URM_all)
    URM_train_validation, validation_data = leave_one_out(URM_train_test)

    return URM_train_validation, URM_train_test, validation_data, test_data


def leave_one_out(URM):
    """Set to 0 ten random elements into train matrix for each designated test user"""

    URM_hold_out = URM.copy()
    # indices = sample_test_users(URM_hold_out, sample_size, sample_threshold=leave_out)

    # Convert to lil matrix to directly access elements in a faster way than CSR
    # URM_hold_out = sparse.lil_matrix(URM_hold_out)

    n_users = URM_hold_out.shape[0]
    selected_users = []
    selected_items = []

    for user_id in range(n_users):

        start_pos = URM_hold_out.indptr[user_id]
        end_pos = URM_hold_out.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            hold_out_item = random.sample(range(start_pos, end_pos), 1)

            selected_users.append(user_id)
            selected_items.append(URM_hold_out.indices[hold_out_item][0])

            URM_hold_out.data[hold_out_item] = 0  # set to 0 in train

    URM_hold_out.eliminate_zeros()
    URM_hold_out = URM_hold_out.tocsr()
    hold_out_data = dict(zip(selected_users, selected_items))

    return URM_hold_out, hold_out_data


