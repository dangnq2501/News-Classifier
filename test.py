from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
from model import BERTClassifier

import pandas as pd
import joblib
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("./local_directory/")
# base_model = joblib.load("base_line.pkl")


# bert_model = BERTClassifier(hidden_size=32, dropout=0.13913256314959768)
df_test = pd.read_csv("data/test.txt", sep="\t", quoting=3, encoding="utf-8")
labels = df_test["Category"].unique()
inv_labels = {}
labels_name = {
    'b': 'business', 
    't': 'science and technology', 
    'e': 'entertainment', 
    'm': 'health'
}
print(list(labels_name.keys()))
# bert_model.load_state_dict(torch.load("bert_model_32.pth", weights_only=True, map_location="cpu"))
# bert_model.eval()
# text = "Stepping Up the Pace Means Leaving Nobody Behind"
# token = tokenizer(
#             text,
#             padding="max_length",
#             max_length=40,
#             truncation=True,
#             return_tensors="pt"
#         )
# output = bert_model(token["input_ids"], token["attention_mask"])
# preds = output.argmax(dim=1)
# label = labels_name[labels[preds.cpu()[0]]]
# proba = F.softmax(output, dim=1)
# print(proba.max().detach().numpy())