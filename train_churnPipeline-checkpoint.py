import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(x_train, y_train):
    cat_feat = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_feat = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    #for impute missing value numerical features and categorical
    numeric_preprocess = Pipeline([('num_imputer', SimpleImputer(strategy='mean'))])
    categorical_preprocess= Pipeline([('cat_imputer', SimpleImputer(strategy='most_frequent')),
                                  ('cat_encoder', OrdinalEncoder(categories=[['Male', 'Female'],
                                                                            ['Basic', 'Standard', 'Premium'],
                                                                            ['Monthly','Quarterly','Annual']]))])
    
    preprocess=ColumnTransformer(transformers=[
        ('numPreprocess', numeric_preprocess, num_feat),
        ('catPreprocess', categorical_preprocess,(['Gender','SubscriptionType','ContractLength']))],
                                 remainder='drop')

    churn_pred = Pipeline([
        ('preprocessing', preprocess),
        ('classifier', RandomForestClassifier(criterion= 'gini',max_depth=4))])
                                  

    # ===== MLflow tracking =====
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run() as run:
        # log parameters
        mlflow.log_param("criterion", "gini")
        mlflow.log_param("max_depth", 4)

        # train
        churn_pred.fit(x_train, y_train)

        # save and log model
        joblib.dump(churn_pred, "artifacts/churn_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(churn_pred,artifact_path="model")

    return run.info.run_id