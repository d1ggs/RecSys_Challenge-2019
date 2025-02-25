#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

from Legacy.Base.BaseRecommender import BaseRecommender
from Legacy.Base.DataIO import DataIO
import numpy as np



class BaseSimilarityMatrixRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self, URM_train):
        super(BaseSimilarityMatrixRecommender, self).__init__(URM_train)

        self._compute_item_score = self._compute_score_item_based

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               COMPUTE ITEM SCORES                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _compute_score_item_based(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()


        item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores)
        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


    def _compute_score_user_based(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train).toarray()

        item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores)
        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


    def _check_format(self):

        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:

            if self.W_sparse.getformat() != "csr":
                print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("W_sparse", "csr"))

            self._W_sparse_format_checked = True



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict_to_save = {"W_sparse": self.W_sparse}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))
