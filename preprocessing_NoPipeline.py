import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import OrdinalEncoder


def load_data():
    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv("ingested/customer_churn.csv",sep=";")

    X = df.drop(['Churn','CustomerID'],axis=1)
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test

def missing_value(x_train,x_test):
    impute_stats = {"tenure_mean": x_train['Tenure'].mean(),
                      "support_calls_mean": x_train['Support Calls'].mean(),
                      "total_spend_mean": x_train['Total Spend'].mean(),
                      "gender_mode": x_train['Gender'].mode()[0]}
    
    x_train['Tenure']= x_train['Tenure'].fillna(impute_stats['tenure_mean'])
    x_train['Support Calls']=x_train['Support Calls'].fillna(impute_stats['support_calls_mean'])
    x_train['Total Spend']=x_train['Total Spend'].fillna(impute_stats['total_spend_mean'])


    x_test['Tenure']=x_test['Tenure'].fillna(impute_stats['tenure_mean'])
    x_test['Support Calls']=x_test['Support Calls'].fillna(impute_stats['support_calls_mean'])
    x_test['Total Spend']=x_test['Total Spend'].fillna(impute_stats['total_spend_mean'])

    x_train['Gender'] = x_train['Gender'].fillna(impute_stats['gender_mode'])
    x_test['Gender']  = x_test['Gender'].fillna(impute_stats['gender_mode'])

    joblib.dump(impute_stats, "artifacts/impute_stats.pkl")
                                                       
    return x_train, x_test

def encoder(x_train,x_test):
    x_train = (x_train.replace({"Gender": {"Male": 1, "Female": 0}}))
    x_test = (x_test.replace({"Gender": {"Male": 1, "Female": 0}}))

    subs_categories = [['Basic', 'Standard', 'Premium']]
    cont_categories = [['Monthly', 'Quarterly', 'Annual']]
    
    subs_enc_train=x_train[['Subscription Type']]
    cont_enc_train=x_train[['Contract Length']]
    
    subs_enc_test=x_test[['Subscription Type']]
    cont_enc_test=x_test[['Contract Length']]
    
    subs_encoder = OrdinalEncoder(categories=subs_categories,handle_unknown='use_encoded_value',unknown_value=-1)
    cont_encoder = OrdinalEncoder(categories=cont_categories,handle_unknown='use_encoded_value',unknown_value=-1)
    
    subs_train_encoded = subs_encoder.fit_transform(subs_enc_train)
    cont_train_encoded = cont_encoder.fit_transform(cont_enc_train)
    
    subs_test_encoded = subs_encoder.transform(subs_enc_test)
    cont_test_encoded = cont_encoder.transform(cont_enc_test)
    
    subs_train_encoded = pd.DataFrame(subs_train_encoded,columns=['Subscription Type Ordinal'])
    cont_train_encoded = pd.DataFrame(cont_train_encoded,columns=['Contract Length Ordinal'])
    subs_test_encoded = pd.DataFrame(subs_test_encoded,columns=['Subscription Type Ordinal'])
    cont_test_encoded = pd.DataFrame(cont_test_encoded,columns=['Contract Length Ordinal'])
    
    
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    
    x_train_enc = pd.concat([x_train, subs_train_encoded, cont_train_encoded], axis=1)
    x_test_enc = pd.concat([x_test, subs_test_encoded, cont_test_encoded], axis=1)

    x_train_enc=x_train_enc.drop(['Subscription Type', 'Contract Length'],axis=1)
    x_test_enc=x_test_enc.drop(['Subscription Type', 'Contract Length'],axis=1)

   
    joblib.dump(subs_encoder, 'artifacts/ordinal_encode_subs.pkl')
    joblib.dump(cont_encoder, 'artifacts/ordinal_encode_cont.pkl')
    
    return x_train_enc, x_test_enc

    

    
    