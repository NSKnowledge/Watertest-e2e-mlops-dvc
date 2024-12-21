import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier



#*********************  MLFlow & Dagshub Setup  **************************************************

dagshub.init(repo_owner='NSKnowledge', repo_name='Watertest-e2e-mlops-dvc', mlflow=True)
mlflow.set_experiment("experiment4_rf")
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


cls= RandomForestClassifier(random_state=42)

# params ={
#     "n_estimators":[100,200,400,500,800,1000],
#     "max-depth":["None",2,4,8,10,20]
# }

params = {
    "n_estimators": [100, 200, 400, 500, 800, 1000],
    "max_depth": [None, 2, 4, 8, 10, 20]  # Corrected parameter name
}

random_search_cv = RandomizedSearchCV(estimator=cls,param_distributions=params,n_iter=50,cv=5,n_jobs=-1, verbose=2, random_state=42)

with mlflow.start_run(run_name=f"hyperparameter_tuning") as parent_run:
    
    random_search_cv.fit(X_train,y_train)  

    for i in range(len(random_search_cv.cv_results_['params'])):
        with mlflow.start_run(run_name=f"combination{i+1}",nested=True) as child_run:
            mlflow.log_param(f"parameters for {i+1}",random_search_cv.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score", random_search_cv.cv_results_['mean_test_score'][i])
    
    print("Best Params: ", random_search_cv.best_params_)

    best_rf = random_search_cv.best_estimator_

    best_rf.fit(X_train,y_train)
    pickle.dump(best_rf,open("notebooks/exp4/model.pkl","wb"))

    y_pred = best_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix")
    plt.savefig(f"notebooks/exp4/confusion_matrix.png")

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1score",f1score)

    mlflow.log_param("best parameters",random_search_cv.best_params_)
    

    mlflow.log_artifact(f"notebooks/exp4/confusion_matrix.png")
    mlflow.sklearn.log_model(random_search_cv.best_estimator_,"Randomforest classifier")

    mlflow.log_artifact(__file__)
    tags ={
        "author" : "NSKnowledge",
        "model"  : "Randomforestclassifier"
    }
    mlflow.set_tags(tags)

    training_data = mlflow.data.from_pandas(train_processed_data)
    test_data = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(training_data,"training_data")
    mlflow.log_input(test_data,"test _data")

    print("accuracy: ",accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1score: ",f1score)


