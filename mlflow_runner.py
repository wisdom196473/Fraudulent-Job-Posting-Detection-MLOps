import logging
import warnings
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import h2o
from h2o.automl import H2OAutoML
h2o.init()

Y_PRED = 'fraudulent'

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(40)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def run_mlflow_pipeline(path: pd.DataFrame):
    df = h2o.upload_file(path)
    
    predictors = df.columns
    predictors.remove(Y_PRED)

    # For binary classification, response should be a factor
    df[Y_PRED] = df[Y_PRED].asfactor()

    train, test = df.split_frame(ratios = [0.8], seed = 1234)

    aml = H2OAutoML(max_models=15, seed=1, include_algos=["XGBoost", "DRF", "DeepLearning"], 
                    sort_metric="AUCPR", balance_classes=True, nfolds=5) #area under precision recall curve
    aml.train(x=predictors, y=Y_PRED, training_frame=train)
    
    best_model = aml.leader
    second_best = h2o.get_model(aml.leaderboard.as_data_frame()["model_id"][1])
    
    test_df = test.as_data_frame()
    test_path = "test_set.csv"
    test_df.to_csv(test_path, index=False)
    
    models = [{'model':best_model, 'name':'Leader_Model_Run'}, 
              {'model':second_best, 'name': 'Second_Best_Run'}]
    
    for model_config in models:
        name = model_config['name']
        model = model_config['model']
        path = name.split('_')[0]
        with mlflow.start_run(run_name=name):
            mlflow.log_metric(f"{model.model_id}_logloss", model.logloss(xval=True))
            mlflow.log_metric(f"{model.model_id}_auc", model.auc(xval=True))
            mlflow.log_metric(f"{model.model_id}_aucpr", model.aucpr(xval=True))
            
            #Focus on the performance of where fraudulent is true in the results. Take that value for each metric
            mlflow.log_metric(f"{model.model_id}_accuracy", model.accuracy(xval=True)[0][1])
            mlflow.log_metric(f"{model.model_id}_recall", model.recall(xval=True)[0][1])
            mlflow.log_metric(f"{model.model_id}_F1", model.F1(xval=True)[0][1])
        
            mlflow.log_artifact(test_path, artifact_path="test_data")
            
            signature = infer_signature(train.as_data_frame(), model.predict(train).as_data_frame())
            mlflow.h2o.log_model(model, artifact_path=path, signature=signature)
