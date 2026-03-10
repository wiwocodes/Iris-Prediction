import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('artifacts/model_churnNoPipeline.pkl')
impute_stats=joblib.load('artifacts/impute_stats.pkl')
ordinal_encode_subs=joblib.load('artifacts/ordinal_encode_subs.pkl')
ordinal_encode_cont=joblib.load('artifacts/ordinal_encode_cont.pkl')


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
    df = (df.replace({"Gender": {"Male": 1, "Female": 0}}))

    df['Tenure']=df['Tenure'].fillna(impute_stats['tenure_mean'])
    df['Support Calls']=df['Support Calls'].fillna(impute_stats['support_calls_mean'])
    df['Total Spend']=df['Total Spend'].fillna(impute_stats['total_spend_mean'])
    df['Gender'] = df['Gender'].fillna(impute_stats['gender_mode'])
    
    cat_subs=df[['Subscription Type']]
    cat_cont=df[['Contract Length']]
    cat_enc_subs=pd.DataFrame(ordinal_encode_subs.transform(cat_subs),columns=['Subscription Type Ordinal'])
    cat_enc_cont=pd.DataFrame(ordinal_encode_cont.transform(cat_cont),columns=['Contract Length Ordinal'])
    df=pd.concat([df,cat_enc_subs,cat_enc_cont], axis=1)
    df=df.drop(['Subscription Type', 'Contract Length'],axis=1)
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
