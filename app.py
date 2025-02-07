from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn

app = FastAPI()

import joblib
model = joblib.load("text_classifier.pkl")

labels = model.named_steps["clf"].classes_.tolist()

class TextInput(BaseModel):
    text: str

@app.get("/list_label")
def list_label():
    return {"labels": labels}

@app.post("/classify")
def classify_text(input_text: TextInput):
    prediction_proba = model.predict_proba([input_text.text])
    predicted_label = labels[prediction_proba.argmax()]
    confidence = prediction_proba.max()

    return {
        "text": input_text.text,
        "predicted_label": predicted_label,
        "prob": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2005)
