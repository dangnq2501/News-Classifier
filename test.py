from transformers import AutoTokenizer, AutoModel
from model import LSTM
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

tokenizer = AutoTokenizer.from_pretrained("./local_directory/")
model = AutoModel.from_pretrained("./local_directory/")
def get_embed(x):
    return tokenizer.encode(x, padding="max_length", max_length=40, truncation=True)
def len_embed(x):
    return len(x)
def prepare(df):
    df["Embed"] = df["Title"].apply(get_embed)
    df["Length"] = df["Embed"].apply(len_embed)
    # print(df["Length"].min(), df["Length"].max())
    df["Label"] = df["Category"].apply(lambda x : inv_labels[x])
    # print(df["Embed"].head(5))    
df_train = pd.read_csv("data/test.txt", sep="\t", quoting=3, encoding="utf-8")
labels = df_train["Category"].unique()
inv_labels = {}
for i in range(len(labels)):
    inv_labels[labels[i]] = i
prepare(df_train)
dataset = TensorDataset(torch.tensor(df_train["Embed"]), torch.tensor(df_train["Label"]))
dataloader = DataLoader(dataset, batch_size=128)
for data in dataloader:
    X, y = data 
    X = X.unsqueeze(-1)
    print(X.shape, y.shape)