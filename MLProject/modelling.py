# modelling.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="diabetes_preprocessing.csv")
args = parser.parse_args()

mlflow.set_experiment("diabetes_ugisugih_experiment")

DATA_PATH = args.data_path
TARGET = "Outcome"

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def main():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.xgboost.autolog()

    with mlflow.start_run(run_name="xgb_autolog_run"):
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)

        print("Metrics: AUC=%.4f, F1=%.4f, Acc=%.4f" % (auc, f1, acc))
        mlflow.log_metric("roc_auc_manual", auc)

        model_path = "artifacts/xgb_model.joblib"
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model_artifacts")

if __name__ == "__main__":
    main()
