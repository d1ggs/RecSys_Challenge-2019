import pandas as pd
import os
ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    train_matrix = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/dataset.csv"))
    user_list = pd.Series(train_matrix.row.unique())
    user_list.columns = ["user_id"]
    user_list.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/target_users.csv"), index=False, header=True)