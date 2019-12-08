from tqdm import trange

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from UserBasedCBF import UserBasedCBF
from ItemBasedCBF import ItemBasedCBF
from ItemCollaborativeFilter import ItemCollaborativeFilter
from UserCollaborativeFilter import UserCollaborativeFilter
from SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from TopPopularRecommender import TopPopRecommender

from utils.helper import Helper
import numpy as np
from utils.run import RunRecommender

if __name__ == '__main__':
    helper = Helper()
    URM_train = helper.URM_train_validation

    # print(helper.URM_train_validation.shape)
    # print(helper.URM_train_test.shape)
    # print(len(helper.validation_data.keys()))
    # print(len(helper.test_data.keys()))
    #
    # exit()

    profile_length = np.ediff1d(URM_train.indptr)

    SLIM_params = {'alpha': 0.0023512567548654,
                   'l1_ratio': 0.0004093694334328875,
                   'positive_only': True,
                   'topK': 25}
    user_cf_parameters = {"topK": 410,
                          "shrink": 0}
    item_cf_parameters = {"topK": 29,
                          "shrink": 22}
    user_cbf_parameters = {"topK_age": 48,
                           "topK_region": 0,
                           "shrink_age": 8,
                           "shrink_region": 1,
                           "age_weight": 0.3}

    slim = MultiThreadSLIM_ElasticNet(URM_train)
    user_cbf = UserBasedCBF(URM_train)
    item_cbf = ItemBasedCBF(URM_train)
    item_based = ItemCollaborativeFilter(URM_train)
    user_based = UserCollaborativeFilter(URM_train)
    top_popular = TopPopRecommender(URM_train)

    slim.fit(**SLIM_params)
    user_cbf.fit(**user_cbf_parameters)
    item_cbf.fit()
    item_based.fit(**item_cf_parameters)
    user_based.fit(**user_cf_parameters)
    top_popular.fit()

    block_size = int(len(profile_length) * 0.05)
    sorted_users = np.argsort(profile_length)

    ok_users = set(helper.validation_data.keys())

    approved_users = []

    for user in sorted_users:
        if user in ok_users:
            approved_users.append(user)
    sorted_users = np.array(approved_users)

    MAP_slim_per_group = []
    MAP_item_based_CBF_per_group = []
    MAP_user_based_CBF_per_group = []
    MAP_item_cf_per_group = []
    MAP_user_cf_per_group = []
    MAP_toppop_per_group = []
    cutoff = 10
    groups = []

    for group_id in range(0, 10):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("########################## Group", group_id, "##########################")

        groups.append("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        users_in_group = set(list(users_in_group.astype(int)))
        print("SLIM ElasticNet")
        MAP_slim_per_group.append(RunRecommender.evaluate_profile(slim, users_to_evaluate=users_in_group))
        print("Item Based CF")
        MAP_item_cf_per_group.append(RunRecommender.evaluate_profile(item_based, users_to_evaluate=users_in_group))
        print("User Based CF")
        MAP_user_cf_per_group.append(RunRecommender.evaluate_profile(user_based, users_to_evaluate=users_in_group))
        print("User Based CBF")
        MAP_user_based_CBF_per_group.append(RunRecommender.evaluate_profile(user_cbf, users_to_evaluate=users_in_group))
        print("Item Based CBF")
        MAP_item_based_CBF_per_group.append(RunRecommender.evaluate_profile(item_cbf, users_to_evaluate=users_in_group))
        print("Top Popular")
        MAP_toppop_per_group.append(RunRecommender.evaluate_profile(top_popular, users_to_evaluate=users_in_group))


    import matplotlib.pyplot as pyplot

    pyplot.plot(MAP_slim_per_group, label="SLIM")
    pyplot.plot(MAP_item_based_CBF_per_group, label="Item CBF")
    pyplot.plot(MAP_user_based_CBF_per_group, label="User CBF")
    pyplot.plot(MAP_user_cf_per_group, label="User CF")
    pyplot.plot(MAP_item_cf_per_group, label="Item CF")
    pyplot.plot(MAP_toppop_per_group, label="TopPop")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()

    for group in groups:
        print(group)



