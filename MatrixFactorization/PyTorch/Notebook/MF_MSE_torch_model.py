import torch

class MF_MSE_PyTorch_model(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors):

        super(MF_MSE_PyTorch_model, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.user_factors = torch.nn.Embedding(num_embeddings = self.n_users, embedding_dim = self.n_factors)
        self.item_factors = torch.nn.Embedding(num_embeddings = self.n_items, embedding_dim = self.n_factors)

        self.layer_1 = torch.nn.Linear(in_features = self.n_factors, out_features = 1)

        self.activation_function = torch.nn.ReLU()



    def forward(self, user_coordinates, item_coordinates):

        current_user_factors = self.user_factors(user_coordinates)
        current_item_factors = self.item_factors(item_coordinates)

        prediction = torch.mul(current_user_factors, current_item_factors)

        prediction = self.layer_1(prediction)
        prediction = self.activation_function(prediction)

        return prediction



    def get_W(self):

        return self.user_factors.weight.detach().cpu().numpy()


    def get_H(self):

        return self.item_factors.weight.detach().cpu().numpy()