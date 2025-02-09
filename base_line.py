import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import optuna

df_train = pd.read_csv("data/train.txt", sep="\t", quoting=3, encoding="utf-8")
df_val = pd.read_csv("data/valid.txt", sep="\t", quoting=3, encoding="utf-8")
X_train, y_train = df_train["Title"], df_train["Category"]
X_valid, y_valid = df_val["Title"], df_val["Category"]

def objective(trial):
    C = trial.suggest_float("C", 0.01, 10) 
    max_features = trial.suggest_int("max_features", 100, 10000)
    max_iter = trial.suggest_int("max_iter", 500, 5000)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("clf", LogisticRegression(C=C, max_iter=max_iter))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    f1_macro = f1_score(y_valid, y_pred, average="macro")
    return f1_macro 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_C = study.best_params["C"]
best_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=study.best_params["max_features"])),
    ("clf", LogisticRegression(C=best_C, max_iter=study.best_params["max_iter"]))
])
best_pipeline.fit(X_train, y_train)

print("Best C:", best_C)
joblib.dump(best_pipeline, "regression.pkl") 
