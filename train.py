from transformers import AutoTokenizer, AutoModel
from model import LSTM
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna

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
df_train = pd.read_csv("data/train.txt", sep="\t", quoting=3, encoding="utf-8")
df_val = pd.read_csv("data/valid.txt", sep="\t", quoting=3, encoding="utf-8")
df_test = pd.read_csv("data/test.txt", sep="\t", quoting=3, encoding="utf-8")
labels = df_train["Category"].unique()
inv_labels = {}
for i in range(len(labels)):
    inv_labels[labels[i]] = i

prepare(df_train)
prepare(df_val)
prepare(df_test)
# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#       self.X = X
#       self.y = y
    
#     def __len__(self):
#       return len(self.X)

#     def __getitem__(self, idx):
#        return {self.X[idx], self.y[idx]}
    
def get_dataloader(df, batch_size=64, shuffle=False):
   dataset = TensorDataset(torch.tensor(df["Embed"]), torch.tensor(df["Label"]))
   return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
# print(df_train.columns)
# X_train, y_train = df_train["Title"], df_train["Category"]
batch_size= 128
train_dataloader = get_dataloader(df_train, batch_size, True)
val_dataloader = get_dataloader(df_val, batch_size, True)
test_dataloader = get_dataloader(df_test, batch_size)


def train_model(trial, train_loader, val_loader, device):

    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, bias=True, output_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(20):  
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch, y_batch
            X_batch = X_batch.float().unsqueeze(-1)
            y_batch = y_batch.long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch, y_batch
            X_batch = X_batch.float().unsqueeze(-1)
            y_batch = y_batch.long()            
            preds = model(X_batch).argmax(dim=1)
            y_preds.extend(preds)
            y_true.extend(y_batch)
    return accuracy_score(y_true, y_preds)

def evaluate(param, name_model, train_loader, test_loader, device="cpu"):
  model = LSTM(input_size=1, hidden_size=param["hidden_size"], num_layers=param["num_layers"], bias=True, output_size=4)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=param["lr"])
  scheduler = CosineAnnealingLR(optimizer, T_max=10)


  for epoch in range(30):  
      for X_batch, y_batch in train_loader:
          X_batch, y_batch = X_batch, y_batch
          X_batch = X_batch.float().unsqueeze(-1)
          y_batch = y_batch.long()
          optimizer.zero_grad()
          outputs = model(X_batch)
          loss = criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()
      scheduler.step()
  model.eval()
  y_preds, y_true = [], []
  with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch, y_batch
        X_batch = X_batch.float().unsqueeze(-1)
        y_batch = y_batch.long()            
        preds = model(X_batch).argmax(dim=1)
        y_preds.extend(preds)
        y_true.extend(y_batch)
    cm = confusion_matrix(y_true.data, y_preds.cpu())
    # print(pred)
    ConfusionMatrixDisplay(cm).plot()
    print("Accuracy: ", accuracy_score(y_true, y_preds))
  joblib.dump(model, f'lstm.pkl')

model = LSTM(input_size=1, hidden_size=256, num_layers=4,bias=True, output_size=4)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs =30
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: train_model(trial, train_dataloader, val_dataloader, device="cpu"), n_trials=10)

print("Best Hyperparameters:", study.best_params)
evaluate(model, 'lstm', train_dataloader, test_dataloader)
