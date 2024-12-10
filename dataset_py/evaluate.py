import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from model import *
from utils import *
import warnings
import json
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="GNN-based Recommendation System with User CSV")
    parser.add_argument("--data_path", type=str, default="dataset/preprocessed_data.pt", help="Path to preprocessed data")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to trained model")
    parser.add_argument("--item_csv", type=str, default="dataset/item.csv", help="Path to item metadata CSV file")
    parser.add_argument("--user_csv", type=str, default="dataset/new_user.csv", help="Path to new user CSV file")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top recommendations")
    parser.add_argument("--top_n_closest", type=int, default=3, help="Number of closest items based on distance")
    return parser.parse_args()

def embed_new_user(new_user_text, tokenizer, bert_model):
    inputs = tokenizer(new_user_text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        text_embedding = bert_model(**inputs).last_hidden_state[:, 0, :]
    return text_embedding

def add_new_user_and_calculate_scores(new_user_embedding, model, train_data):
    updated_data = train_data
    updated_user_features = torch.cat([updated_data['user'].x, new_user_embedding], dim=0)
    updated_data['user'].x = updated_user_features
    new_user_index = updated_user_features.shape[0] - 1
    model.eval()
    with torch.no_grad():
        x_dict = {'user': updated_data['user'].x, 'item': updated_data['item'].x}
        edge_index_dict = {
            ('user', 'interacts', 'item'): updated_data['user', 'interacts', 'item'].edge_index,
            ('item', 'rev_interacts', 'user'): updated_data['user', 'interacts', 'item'].edge_index.flip(0)
        }
        out = model(x_dict, edge_index_dict)
    user_embedding = out['user'][new_user_index]
    item_embeddings = out['item']
    scores = torch.matmul(user_embedding, item_embeddings.T)
    return scores, item_embeddings

def recommend_top_n_items(scores, item_id_map, N=10):
    top_n_indices = torch.topk(scores, N).indices
    item_id_reverse_map = {v: k for k, v in item_id_map.items()}
    top_n_items = [item_id_reverse_map[idx.item()] for idx in top_n_indices]
    return top_n_items

def calculate_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def find_closest_paths(user_lat, user_lon, item_df, top_n=3):
    item_df['distance'] = item_df.apply(
        lambda row: calculate_distance(user_lat, user_lon, row['위도'], row['경도']), axis=1
    )
    closest_paths = item_df.nsmallest(top_n, 'distance')[['위도', '경도', '걷기코스 구분명', '걷기코스 이름' ,'distance']]
    closest_paths = closest_paths[[ '걷기코스 구분명', '걷기코스 이름' ,'distance']]
    return closest_paths

def create_user_texts(user_csv_path, user_text_columns):
    df = pd.read_csv(user_csv_path)
    user_texts = df[user_text_columns].apply(lambda row: ','.join(map(str, row.values)), axis=1)
    return user_texts.tolist()

def extract_indices_from_top_n_items(top_n_items):
    indices = [int(item.split('_')[1]) for item in top_n_items]
    return indices

def create_new_item_df(item_df, indices):
    return item_df.iloc[indices]

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    bert_model = AutoModel.from_pretrained("klue/bert-base")
    user_text_columns = ["성별", "키", "몸무게", "거주지역", "선호하는 장소", "트래킹 난이도", "위도", "경도"]
    user_texts = create_user_texts(args.user_csv, user_text_columns)
    new_user_text = user_texts[0]

    raw_new_user_embedding = embed_new_user(new_user_text, tokenizer, bert_model)
    user_features, item_features, train_df, val_df, test_df, user_id_map, item_id_map = load_preprocessed_data("dataset/preprocessed_data.pt")

    train_df = train_df.dropna().reset_index(drop=True)
    train_data = create_hetero_data(user_features, item_features, train_df)

    model = GNNRecommender(input_dim=768, hidden_dim=256)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    scores, item_embeddings = add_new_user_and_calculate_scores(raw_new_user_embedding, model, train_data)
    top_n_items = recommend_top_n_items(scores, item_id_map, args.top_n)
    indices = extract_indices_from_top_n_items(top_n_items)
    item_df = create_new_item_df(pd.read_csv(args.item_csv), indices)

    user_lat, user_lon = map(float, new_user_text.split(',')[-2:])
    top_3_paths = find_closest_paths(user_lat, user_lon, item_df, top_n=args.top_n_closest)
    top_3_paths_json = top_3_paths.to_dict(orient='records')
    with open("top_3_paths.json", "w", encoding="utf-8") as json_file:
        json.dump(top_3_paths_json, json_file, ensure_ascii=False, indent=4)

    print("Top-3 closest paths have been saved to 'top_3_paths.json'.")

if __name__ == "__main__":
    main()