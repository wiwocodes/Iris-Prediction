from data_ingestion import ingest_data
from preprocessing_NoPipeline import (load_data,missing_value,encoder)
from train_churnNoPipeline import train
from evaluation_NoPipeline import evaluate

ACCURACY_THRESHOLD = 0.9

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    print("Step 2: Preprocessing")
    x_train, x_test, y_train, y_test=load_data()
    x_train, x_test=missing_value(x_train,x_test)
    x_train_enc, x_test_enc=encoder(x_train,x_test)


    print("Step 3: Training")
    run_id = train(x_train_enc, y_train)

    print("Step 4: Evaluation")
    accuracy,precision,recall = evaluate(x_test_enc,y_test,run_id)

    if accuracy >= ACCURACY_THRESHOLD:
        print("Model approved for deployment")
    else:
        print("Model rejected")

if __name__ == "__main__":
    run_pipeline()

