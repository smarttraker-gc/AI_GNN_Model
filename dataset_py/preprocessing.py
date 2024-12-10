import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

user_df = pd.read_csv("dataset/user.csv")
item_df = pd.read_csv("dataset/item.csv")
interaction_df = pd.read_csv("dataset/preferences.csv")

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModel.from_pretrained("klue/bert-base")

def embed_texts(texts, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Texts"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(embeddings, dim=0)

def add_column_names_to_values(df):
    return df.apply(lambda row: " ".join([f"{col}:{row[col]}" for col in df.columns]), axis=1)

user_df["사용자 ID"] = ["사용자_" + str(idx) for idx in range(len(user_df))]
user_text_columns = ["성별", "키", "몸무게", "거주지역", "선호하는 장소", "트래킹 난이도", "위도", "경도"]
user_texts = add_column_names_to_values(user_df[user_text_columns]).tolist()
user_features_tensor = embed_texts(user_texts)

item_df["산책로 ID"] = ["산책로_" + str(idx) for idx in range(len(item_df))]
item_text_columns = ["행정구역명", "코스 난이도", "코스 경관 카테고리", "소요시간", "주소", "위도", "경도"]
item_texts = add_column_names_to_values(item_df[item_text_columns]).tolist()
item_features_tensor = embed_texts(item_texts)


user_id_map = {uid: idx for idx, uid in enumerate(user_df["사용자 ID"].unique())}
item_id_map = {iid: idx for idx, iid in enumerate(item_df["산책로 ID"].unique())}
interaction_df["user_numeric_id"] = interaction_df["사용자 ID"].map(user_id_map)
interaction_df["item_numeric_id"] = interaction_df["산책로 ID"].map(item_id_map)

random_seed = 42
train_df, test_df = train_test_split(interaction_df, test_size=0.2, random_state=random_seed)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=random_seed)

torch.save({
    "user_features": user_features_tensor,
    "item_features": item_features_tensor,
    "train": train_df,
    "val": val_df,
    "test": test_df,
    "user_id_map": user_id_map,
    "item_id_map": item_id_map,
}, "dataset/preprocessed_data.pt")
print("Preprocessed data saved successfully!")
