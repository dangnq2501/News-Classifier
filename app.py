from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
from model import BERTClassifier
import joblib
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("./local_directory/")
app = FastAPI()


base_model = joblib.load("base_line.pkl")
labels = base_model.named_steps["clf"].classes_.tolist()

bert_model = BERTClassifier(hidden_size=32, dropout=0.13913256314959768)
bert_model.load_state_dict(torch.load("bert_model_32.pth", weights_only=True, map_location="cpu"))
bert_model.eval()
df_test = pd.read_csv("data/test.txt", sep="\t", quoting=3, encoding="utf-8")
labels = df_test["Category"].unique()
inv_labels = {}
labels_name = {
    'b': 'business', 
    't': 'science and technology', 
    'e': 'entertainment', 
    'm': 'health'
}
class TextInput(BaseModel):
    text: str

@app.get("/list_label")
def list_label():
    return {"labels": list(labels_name.values())}

@app.post("/classify")
def classify_text(input_text: TextInput):
    # prediction_proba = base_model.predict_proba([input_text.text])
    # predicted_label = labels[prediction_proba.argmax()]
    # confidence = prediction_proba.max()

    token = tokenizer(
                input_text.text,
                padding="max_length",
                max_length=40,
                truncation=True,
                return_tensors="pt"
            )
    output = bert_model(token["input_ids"], token["attention_mask"])
    preds = output.argmax(dim=1)
    predicted_label = labels_name[labels[preds.cpu()[0]]]
    confidence = F.softmax(output, dim=1).max().detach().numpy()
    # print(confidence)
    return {
        "text": input_text.text,
        "predicted_label": predicted_label,
        "prob": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2005)
    #uvicorn app:app --reload

