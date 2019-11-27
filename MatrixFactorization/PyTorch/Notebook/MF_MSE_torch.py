from MatrixFactorization.PyTorch.Notebook.MF_MSE_torch_model import MF_MSE_PyTorch_model
import torch

class MF_MSE_torch():
    def __init__(self, URM_train, num_factors=10):
        self.num_factors = num_factors
        self.n_users, self.n_items = URM_train.shape

        self.user_factors = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=num_factors)
        self.item_factors = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=num_factors)

        self.layer_1 = torch.nn.Linear(in_features=num_factors, out_features=1)
        self.activation_function = torch.nn.ReLU()