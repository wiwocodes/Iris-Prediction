from data_ingestion import ingest_data
from train_churnPipeline import train_model
from evaluation_Pipeline import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd

ACCURACY_THRESHOLD = 0.9
 

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()
    
    df = pd.read_csv("ingested/customer_churn.csv",sep=";")
    df=df.rename(columns={"Usage Frequency": "UsageFrequency", "Support Calls": "SupportCalls", 
                   "Payment Delay": "PaymentDelay","Subscription Type":"SubscriptionType",
                   "Contract Length": "ContractLength" , "Total Spend":  "TotalSpend", 
                   "Last Interaction":"LastInteraction" })

    X = df.drop(['Churn','CustomerID'],axis=1)
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("Step 2: Training")
    run_id = train_model(x_train, y_train)

    print("Step 4: Evaluation")
    accuracy,precision,recall = evaluate(x_test,y_test,run_id)

    if accuracy >= ACCURACY_THRESHOLD:
        print("Model approved for deployment")
    else:
        print("Model rejected")

if __name__ == "__main__":
    run_pipeline()

