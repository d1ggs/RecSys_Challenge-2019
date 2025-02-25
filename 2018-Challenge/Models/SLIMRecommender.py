import numpy as np
import time
from tqdm import trange
from scipy import sparse as sps
from Legacy.Base.Recommender_utils import check_matrix


class SLIMRecommender():
    def __init__(self, URM, learning_rate=1e-3):

        self.learning_rate = learning_rate

        self.URM = URM

        self.URM_mask = self.URM.copy()

        # Highlight positive interactions
        # self.URM_mask.data[self.URM.data <= 3] = 0

        self.URM_mask.eliminate_zeros()

        self.n_users = self.URM_mask.shape[0]
        self.n_items = self.URM_mask.shape[1]

        self.eligible_users = []

        for user_id in range(self.n_users):

            start_pos = self.URM_mask.indptr[user_id]
            end_pos = self.URM_mask.indptr[user_id + 1]

            # Find if the user contains any positive interactions, exploiting CSR notation

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligible_users.append(user_id)

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def sample_triplet(self):
        # Select a random user_id
        # TODO check if there are users without interations

        user_seen_items = []
        user_id = 0

        while len(user_seen_items) == 0:
            user_id = np.random.choice(self.URM.shape[0])
            user_seen_items = self.URM_mask[user_id, :].indices

        #print (user_id, user_seen_items)
        pos_item_id = np.random.choice(user_seen_items)

        # Check if it is a negative item

        neg_item_selected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while not neg_item_selected:
            neg_item_id = np.random.randint(0, self.n_items)

            if neg_item_id not in user_seen_items:
                neg_item_selected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):

        learning_rate = 1e-3

        # Get number of available interactions
        num_positive_interactions = int(self.URM_mask.nnz * 0.01)

        # start_time_epoch = time.time()
        # start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(num_positive_interactions):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sample_triplet()

            user_seen_items = self.URM_mask[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, user_seen_items].sum()
            x_j = self.similarity_matrix[negative_item_id, user_seen_items].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update
            self.similarity_matrix[positive_item_id, user_seen_items] += learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, user_seen_items] -= learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            # if time.time() - start_time_batch >= 30 or num_sample == num_positive_interactions - 1:
            #     print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
            #         num_sample,
            #         100.0 * float(num_sample) / num_positive_interactions,
            #         time.time() - start_time_batch,
            #         float(num_sample) / (time.time() - start_time_epoch)))
            #
            #     start_time_batch = time.time()

    def similarityMatrixTopK(self, item_weights, force_sparse_output=True, k=100, verbose=False, inplace=True):
        """
        The function selects the TopK most similar elements, column-wise

        :param item_weights:
        :param force_sparse_output:
        :param k:
        :param verbose:
        :param inplace: Default True, WARNING matrix will be modified
        :return:
        """

        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        start_time = time.time()

        if verbose:
            print("Generating topK matrix")

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        if not sparse_weights:

            print("Sorting columns...")
            idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column
            print("Done!")

            if inplace:
                W = item_weights
            else:
                W = item_weights.copy()

            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            W[not_top_k, np.arange(nitems)] = 0.0

            if force_sparse_output:
                if verbose:
                    print("Starting CSR compression...")

                W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

                if verbose:
                    print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

                return W_sparse

            if verbose:
                print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W

        else:
            # iterate over each column and keep only the top-k similar items
            data, rows_indices, cols_indptr = [], [], []

            item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

            for item_idx in range(nitems):
                cols_indptr.append(len(data))

                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx + 1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

                non_zero_data = column_data != 0

                idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
                top_k_idx = idx_sorted[-k:]

                data.extend(column_data[non_zero_data][top_k_idx])
                rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

            cols_indptr.append(len(data))

            # During testing CSR is faster

            if verbose:
                print("Generating CSC matrix...")

            W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)

            if verbose:
                print("Converting to CSR...")

            W_sparse = W_sparse.tocsr()

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

    def fit(self, learning_rate=0.01, epochs=10):

        self.learning_rate = learning_rate

        for numEpoch in trange(epochs):
            # print("Starting epoch " + str(numEpoch))
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = self.similarityMatrixTopK(self.similarity_matrix, k=100)

    def recommend(self, user_id, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]




