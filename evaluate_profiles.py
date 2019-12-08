from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from UserBasedCBF import UserBasedCBF
from ItemBasedCBF import ItemBasedCBF
from ItemCollaborativeFilter import ItemCollaborativeFilter
from UserCollaborativeFilter import UserCollaborativeFilter

from utils.helper import Helper
import numpy as np
from utils.run import RunRecommender

if __name__ == '__main__':
    helper = Helper()
    URM_train = helper.URM_train_test

    profile_length = np.ediff1d(URM_train.indptr)

    slim = SLIM_BPR_Cython
    user_cbf = UserBasedCBF
    item_cbf = ItemBasedCBF
    item_based = ItemCollaborativeFilter
    user_based = UserCollaborativeFilter

    block_size = int(len(profile_length) * 0.05)
    sorted_users = np.argsort(profile_length)

    MAP_slim_per_group = []
    MAP_item_based_CBF_per_group = []
    MAP_user_based_CBF_per_group = []
    MAP_item_cf_per_group = []
    MAP_user_cf_per_group = []
    cutoff = 10

    for group_id in range(0, 10):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        users_in_group = set(list(users_in_group.astype(int)))

        MAP_slim_per_group.append(RunRecommender.evaluate_on_test_set(SLIM_BPR_Cython, {}, exclude_users=users_in_group))
        MAP_item_cf_per_group.append(RunRecommender.evaluate_on_test_set(ItemCollaborativeFilter, {}, exclude_users=users_in_group))
        MAP_user_cf_per_group.append(RunRecommender.evaluate_on_test_set(UserCollaborativeFilter, {}, exclude_users=users_in_group))
        MAP_user_based_CBF_per_group.append(RunRecommender.evaluate_on_test_set(UserBasedCBF, {}, exclude_users=users_in_group))
        MAP_item_based_CBF_per_group.append(RunRecommender.evaluate_on_test_set(ItemBasedCBF, {}, exclude_users=users_in_group))


    import matplotlib.pyplot as pyplot

    pyplot.plot(MAP_slim_per_group, label="SLIM")
    pyplot.plot(MAP_item_based_CBF_per_group, label="Item CBF")
    pyplot.plot(MAP_user_based_CBF_per_group, label="User CBF")
    pyplot.plot(MAP_user_cf_per_group, label="User CF")
    pyplot.plot(MAP_item_cf_per_group, label="Item CF")
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()



