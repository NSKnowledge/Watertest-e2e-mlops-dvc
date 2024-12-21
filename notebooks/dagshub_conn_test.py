import mlflow

import dagshub

mlflow.set_tracking_uri("https://dagshub.com/NSKnowledge/Watertest-e2e-mlops-dvc.mlflow")


dagshub.init(repo_owner='NSKnowledge', repo_name='Watertest-e2e-mlops-dvc', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', '10')
  mlflow.log_metric('metric name', 1)

