import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn

from model import GNNRecommender  
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = {
        "seed": 42,
        "input_dim": 768,
        "hidden_dim": 256,
        "learning_rate": 0.01,
        "epochs": 100,
        "patience": 100,
        "delta": 0.001,
        "model_path": "best_model.pth",
        "data_path": "dataset/preprocessed_data.pt"
    }

    set_seed(args["seed"])




    user_features, item_features, train_df, val_df, test_df, user_id_map, item_id_map = load_preprocessed_data(args["data_path"])

    train_df = train_df.dropna().reset_index(drop=True)
    val_df = val_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    train_data = create_hetero_data(user_features, item_features, train_df)
    val_data = create_hetero_data(user_features, item_features, val_df)
    test_data = create_hetero_data(user_features, item_features, test_df)

    model = GNNRecommender(input_dim=args["input_dim"], hidden_dim=args["hidden_dim"])
    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])
    loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args["patience"], delta=args["delta"], verbose=True, path=args["model_path"])

    for epoch in range(args["epochs"]):
        model.train()
        optimizer.zero_grad()

        x_dict = {'user': train_data['user'].x, 'item': train_data['item'].x}
        edge_index_dict = {
            ('user', 'interacts', 'item'): train_data['user', 'interacts', 'item'].edge_index,
            ('item', 'rev_interacts', 'user'): train_data['user', 'interacts', 'item'].edge_index.flip(0)
        }

        out = model(x_dict, edge_index_dict)
        user_emb = out['user'][train_data['user', 'interacts', 'item'].edge_index[0]]
        item_emb = out['item'][train_data['user', 'interacts', 'item'].edge_index[1]]

        pred_scores = model.predict(user_emb, item_emb)
        loss = loss_fn(pred_scores, train_data['user', 'interacts', 'item'].y)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x_dict_val = {'user': val_data['user'].x, 'item': val_data['item'].x}
            edge_index_dict_val = {
                ('user', 'interacts', 'item'): val_data['user', 'interacts', 'item'].edge_index,
                ('item', 'rev_interacts', 'user'): val_data['user', 'interacts', 'item'].edge_index.flip(0)
            }

            out_val = model(x_dict_val, edge_index_dict_val)
            user_emb_val = out_val['user'][val_data['user', 'interacts', 'item'].edge_index[0]]
            item_emb_val = out_val['item'][val_data['user', 'interacts', 'item'].edge_index[1]]

            pred_scores_val = model.predict(user_emb_val, item_emb_val)
            val_loss = loss_fn(pred_scores_val, val_data['user', 'interacts', 'item'].y)

            print(f"Epoch {epoch + 1}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.eval()
    with torch.no_grad():
        x_dict_test = {'user': test_data['user'].x, 'item': test_data['item'].x}
        edge_index_dict_test = {
            ('user', 'interacts', 'item'): test_data['user', 'interacts', 'item'].edge_index,
            ('item', 'rev_interacts', 'user'): test_data['user', 'interacts', 'item'].edge_index.flip(0)
        }
        
        out_test = model(x_dict_test, edge_index_dict_test)
        
        user_emb_test = out_test['user'][test_data['user', 'interacts', 'item'].edge_index[0]]
        item_emb_test = out_test['item'][test_data['user', 'interacts', 'item'].edge_index[1]]
        
        pred_scores_test = model.predict(user_emb_test, item_emb_test)
        
        test_loss = loss_fn(pred_scores_test, test_data['user', 'interacts', 'item'].y)
        print(f"Test Loss: {test_loss.item():.4f}")

        model.load_state_dict(torch.load(args["model_path"]))
        print("Loaded the best model.")

if __name__ == "__main__":
    main()
