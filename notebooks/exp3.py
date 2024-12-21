import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


#*********************  MLFlow & Dagshub Setup  **************************************************

dagshub.init(repo_owner='NSKnowledge', repo_name='Watertest-e2e-mlops-dvc', mlflow=True)
mlflow.set_experiment("experiment3_rf")
mlflow.set_tracking_uri("https://dagshub.com/NSKnowledge/Watertest-e2e-mlops-dvc.mlflow")


#*********************  MLFlow SETUP END  **********************************************



data= pd.read_csv(r"C:\Users\abhay\Documents\mlflow\Watertest-e2e-mlops-dvc\DataRepo\water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def fill_missing_values_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
    return df

train_processed_data = fill_missing_values_with_mean(train_data)
test_processed_data = fill_missing_values_with_mean(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle

X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_test =test_processed_data["Potability"]


models={
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machines": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbour": KNeighborsClassifier(),
    "XGBoost": XGBClassifier()
}

n_estimator = 100
max_depth = 10

with mlflow.start_run(run_name=f"trying  various models"):

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):

            #cls= RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=42)

            model.fit(X_train, y_train)
            model_filename = f"{model_name.replace(' ','_')}"

            pickle.dump(model,open(f"notebooks/exp3/{model_filename}.pkl", "wb"))            

            # model = pickle.load(open("notebooks/exp3/model.pkl","rb"))

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1score = f1_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"notebooks/exp3/confusion_matrix_{model_name.replace(" ","_")}.png")

            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1score",f1score)

            mlflow.log_param("n_estimator", n_estimator)
            mlflow.log_param("max_depth", max_depth)

            mlflow.log_artifact(f"notebooks/exp3/confusion_matrix_{model_name.replace(" ","_")}.png")
            mlflow.sklearn.log_model(model, f"{model_name.replace(" ","_")}")

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


