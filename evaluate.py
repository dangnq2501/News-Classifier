import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch 
from sklearn.metrics import classification_report, precision_recall_fscore_support

model = joblib.load("base_line.pkl")
df_test = pd.read_csv("data/test.txt", sep="\t", quoting=3, encoding="utf-8")
X_test, y_true = df_test["Title"], df_test["Category"]

y_pred = model.predict(X_test)        
label_names = y_true.value_counts().index
report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
print(report)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

