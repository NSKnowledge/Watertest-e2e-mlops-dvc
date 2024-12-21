import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub



#*********************  MLFlow & Dagshub Setup  **************************************************

dagshub.init(repo_owner='NSKnowledge', repo_name='Watertest-e2e-mlops-dvc', mlflow=True)
mlflow.set_experiment("experiment1_rf")
mlflow.set_tracking_uri("https://dagshub.com/NSKnowledge/Watertest-e2e-mlops-dvc.mlflow")


#*********************  MLFlow SETUP END  **********************************************



data= pd.read_csv(r"C:\Users\abhay\Documents\mlflow\Watertest-e2e-mlops-dvc\DataRepo\water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def fill_missing_values_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

train_processed_data = fill_missing_values_with_median(train_data)
test_processed_data = fill_missing_values_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle

X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]

n_estimator = 100
max_depth = 10

with mlflow.start_run(run_name=f"n_est={n_estimator}&& maxdepth={max_depth}"):
    cls= RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=42)

    model= cls.fit(X_train, y_train)

    pickle.dump(cls,open("notebooks/exp1/model.pkl", "wb"))

    X_test = test_processed_data.drop(columns=["Potability"], axis=1)
    y_test =test_processed_data["Potability"]

    model = pickle.load(open("notebooks/exp1/model.pkl","rb"))

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("notebooks/exp1/confusion_matrix.png")

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1score",f1score)

    mlflow.log_param("n_estimator", n_estimator)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_artifact("notebooks/exp1/confusion_matrix.png")
    mlflow.sklearn.log_model(cls, "RandomForestClassifier")

    mlflow.log_artifact(__file__)
    tags ={
        "author" : "NSKnowledge",
        "model"  : "Randomforestclassifier"
    }
    mlflow.set_tags(tags)

    print("accuracy: ",accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1score: ",f1score)


