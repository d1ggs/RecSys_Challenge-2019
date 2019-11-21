import numpy as np
from scipy import sparse
import copy


def sample_test_users(URM_csr, sample_size, sample_threshold=10):
    """Sample test user ids of users that have more interactions than a given threshold"""

    candidates = URM_csr.sum(1)
    indices = np.argwhere(candidates > sample_threshold)[:, 0]
    if sample_size > indices.shape[0]:
        print("The sample requested is larger than the candidate users number. Returning whole vector.")
        return indices
    return np.random.choice(indices, sample_size, replace=False)


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row + 1]] = value


def split_train_test(URM_all, split_fraction):
    print("Splitting data into train and test...")
    sample_size = int((1 - split_fraction) * URM_all.shape[0])
    indices = sample_test_users(URM_all, sample_size)

    URM_test = copy.deepcopy(URM_all)
    URM_train = copy.deepcopy(URM_all)

    test_data = {}

    URM_train = sparse.lil_matrix(URM_train)

    for i in range(URM_all.shape[0]):
        if i in indices:
            row = URM_all[i, :]
            interactions = np.argwhere(row > 0)
            np.random.shuffle(interactions)

            test_data[i] = interactions[:, 1][:10]
            for inter in interactions[:10]:
                URM_train[i, inter] = 0
        else:
            csr_row_set_nz_to_val(URM_test, i)

    # URM_train.eliminate_zeros()
    URM_test.eliminate_zeros()

    URM_train = URM_train.tocsr()
    URM_test = URM_test.tocsr()

    # print(URM_train.nnz)
    # print(URM_test.nnz)

    return URM_train, URM_train, indices, test_data
