import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('artifacts/churn_prediction_pipeline.pkl')


def main():
    st.title('Churn Model Deployment')

    # Add user input components for 10 features
    #input one by one
    age=st.number_input("age", 0, 100)
    gender=st.radio("gender", ["Male","Female"])
    tenure=st.number_input("the period of time you holds a position (in years)", 0,100)
    usage_freq=st.number_input("the frequency of product usage (in years)", 0,100)
    support_call=st.number_input("number of support calls", 0,10)
    payment_delay=st.number_input("the period of payment delay (in months)", 0,30)
    subs_type=st.radio("choose subscription type", ["Standard","Premium","Basic"])
    contract_length=st.radio("choose contract length", ["Annual","Quarterly","Monthly"])
    total_spend=st.number_input("total spend in a month", 0,1000000000)
    last_interaction=st.number_input("last interaction with the product (in months)", 0,30)
    
    
    data = {'Age': int(age), 'Gender': gender, 'Tenure':int(tenure),'Usage Frequency':int(usage_freq),
            'Support Calls': int(support_call), 'Payment Delay':int(payment_delay),
            'Subscription Type':subs_type, 'Contract Length': contract_length,
            'Total Spend':int(total_spend),'Last Interaction':int(last_interaction)}
    
    df=pd.DataFrame([list(data.values())], columns=['Age','Gender', 'Tenure', 'Usage Frequency','Support Calls', 
                                                'Payment Delay', 'Subscription Type','Contract Length', 
                                                'Total Spend', 'Last Interaction'])

    df=df.rename(columns={"Usage Frequency": "UsageFrequency", "Support Calls": "SupportCalls", 
                   "Payment Delay": "PaymentDelay","Subscription Type":"SubscriptionType",
                   "Contract Length": "ContractLength" , "Total Spend":  "TotalSpend", 
                   "Last Interaction":"LastInteraction" })
    
    if st.button("Make Prediction"):
        prediction = model.predict(df)[0]
        st.success(f"Churn Prediction: {prediction}")


if __name__ == "__main__":
    main()

