import torch
from torch_geometric.data import HeteroData

def load_preprocessed_data(path):
    data = torch.load(path)
    user_features = data["user_features"]
    item_features = data["item_features"]
    train_df = data["train"]
    val_df = data["val"]
    test_df = data["test"]
    user_id_map = data["user_id_map"]
    item_id_map = data["item_id_map"]
    return user_features, item_features, train_df, val_df, test_df, user_id_map, item_id_map


def create_hetero_data(user_features, item_features, df):
    data = HeteroData()
    data['user'].x = user_features
    data['item'].x = item_features
    user_edges = torch.tensor(df["user_numeric_id"].values, dtype=torch.long)
    item_edges = torch.tensor(df["item_numeric_id"].values, dtype=torch.long)
    edge_weights = torch.tensor(df["선호 점수"].values, dtype=torch.float)
    data['user', 'interacts', 'item'].edge_index = torch.stack([user_edges, item_edges])
    data['user', 'interacts', 'item'].edge_attr = edge_weights
    data['user', 'interacts', 'item'].y = edge_weights
    return data



class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
  
        torch.save(model.state_dict(), self.path)
