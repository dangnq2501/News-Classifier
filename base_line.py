import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


df_train = pd.read_csv("data/train.txt", sep="\t", quoting=3, encoding="utf-8")
df_val = pd.read_csv("data/valid.txt", sep="\t", quoting=3, encoding="utf-8")
X_train, y_train = df_train["Title"], df_train["Category"]
X_valid, y_valid = df_val["Title"], df_val["Category"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

param_grid = {"clf__C": [0.1, 1, 10]}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best model params:", grid_search.best_params_)
joblib.dump(best_model, "base_line.pkl") 

