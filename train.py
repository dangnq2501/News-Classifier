from transformers import AutoTokenizer, AutoModel
from model import LSTM, GRU
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
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
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=40):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
def get_dataloader(df, batch_size=64, shuffle=False):
   dataset = TensorDataset(torch.tensor(df["Embed"]), torch.tensor(df["Label"]))
   return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
# print(df_train.columns)
# X_train, y_train = df_train["Title"], df_train["Category"]
batch_size= 256
train_dataloader = get_dataloader(df_train, batch_size, True)
val_dataloader = get_dataloader(df_val, batch_size)
test_dataloader = get_dataloader(df_test, batch_size)
def train_model(trial, train_loader, val_loader, device="cpu"):
    hidden_size = trial.suggest_int("hidden_size", 32, 1024)
    num_layers = trial.suggest_int("num_layers", 1, 10)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    model = GRU(input_size =40, hidden_size=hidden_size, output_size=4, num_layers=num_layers, bias=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    patience = 5  
    trial_loss = float("inf")
    trial_acc = 0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(50): 
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = X_batch.float().unsqueeze(1)
            y_batch = y_batch.long()

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        model.eval()
        val_loss = 0
        y_preds, y_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = X_batch.float().unsqueeze(1)
                y_batch = y_batch.long()
                
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                
                y_preds.extend(preds.argmax(dim=1).cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        acc = accuracy_score(y_true, y_preds)

        if val_loss < trial_loss:
            trial_loss = val_loss
            trial_acc = acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_model_state:
        torch.save(best_model_state, f"parameters/best_model_{hidden_size}_{num_layers}_{lr}.pth")

    return trial_acc


def evaluate(param, name_model, test_loader, device="cpu"):
#   model = LSTM(input_size=40, hidden_size=param["hidden_size"], num_layers=param["num_layers"], bias=True, output_size=4)
  model = GRU(input_size =40, hidden_size=param["hidden_size"], output_size=4, num_layers=param["num_layers"], bias=True)
  model.load_state_dict(torch.load(f"parameters/best_model_{param["hidden_size"]}_{param["num_layers"]}_{param["lr"]}.pth"))
  model.eval()
  y_preds, y_true = [], []
  with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch, y_batch
        X_batch = X_batch.float().unsqueeze(1)
        y_batch = y_batch.long()            
        preds = model(X_batch).argmax(dim=1)
        y_preds.extend(preds)
        y_true.extend(y_batch)
    cm = confusion_matrix(y_true, y_preds)
    ConfusionMatrixDisplay(cm).plot()
    print("Accuracy: ", accuracy_score(y_true, y_preds))


study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: train_model(trial, train_dataloader, val_dataloader, device="cpu"), n_trials=50)

print("Best Hyperparameters:", study.best_params)
evaluate(study.best_params, 'gru', test_dataloader)