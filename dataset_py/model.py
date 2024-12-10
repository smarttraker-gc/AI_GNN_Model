import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear
import torch

class GNNRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNRecommender, self).__init__()
        self.user_to_item_conv = GCNConv(input_dim, hidden_dim)
        self.item_to_user_conv = GCNConv(input_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = Linear(hidden_dim * 2, 1)
        self.x_dict = dict()

    def forward(self, x_dict, edge_index_dict):
        item_emb = self.user_to_item_conv(x_dict['item'], edge_index_dict[('user', 'interacts', 'item')])
        item_emb = nn.ReLU()(item_emb)
        user_emb = self.item_to_user_conv(x_dict['user'], edge_index_dict[('item', 'rev_interacts', 'user')])
        user_emb = nn.ReLU()(user_emb)
        item_emb = self.projection(item_emb)
        user_emb = self.projection(user_emb)
        self.x_dict['item'] = item_emb
        self.x_dict['user'] = user_emb
        return self.x_dict

    def get_node_embeddings(self, x_dict, edge_index_dict, node_type='user'):
        self.eval()
        with torch.no_grad():
            x_dict = self.forward(x_dict, edge_index_dict)
            return x_dict[node_type]

    def predict(self, user_emb, item_emb):
        edge_features = torch.cat([user_emb, item_emb], dim=1)
        return self.predictor(edge_features).squeeze()
