import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings ('ignore')
import streamlit as st 
import joblib
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st


data= pd.read_csv('expresso_processednew.csv')

df = data.copy()

encoder = LabelEncoder()
scaler = StandardScaler()

df.drop(['Unnamed: 0', 'MRG'], axis = 1, inplace = True)

for i in df.drop('CHURN', axis = 1).columns:
    if df[i].dtypes == 'O':
        df[i] = encoder.fit_transform(df[i])
    else:
        df[i] = scaler.fit_transform(df[[i]])

x = df.drop('CHURN', axis = 1)
y = df.CHURN

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.20, stratify= y)

model = LogisticRegression()
model.fit(xtrain, ytrain)







st.markdown("<h1 style = 'color: #1F4172; text-align: center; font-family: helvetica '>EXPRESSO CHURN PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Ziyah</h4>", unsafe_allow_html = True)
st.image('pngwing.com (2).png', width = 350, use_column_width = True )
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h4 style = 'color: #1F4172; text-align: center; font-family:cursive '>Project Overview</h4>", unsafe_allow_html = True)
st.markdown("<p style = 'text-align: justify'>The predictive telecommunications customer attrition modeling project aims to leverage machine learning techniques to develop an accurate and robust model capable of predicting the whether a customer attricts or not. By analyzing historical data, identifying key features influencing customer's descision, and employing advanced classification algorithms, the project seeks to provide valuable insights for business analysts, entrepreneur, large and small scale businesses. The primary objective of this project is to create a reliable machine learning model that accurately predicts customer's decision based on relevant features such as location, income in ($), client's duration, and other influencing factors. The model should be versatile enough to adapt to different business plans, providing meaningful predictions for a wide range of businesses.", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html = True)
st.dataframe(data, use_container_width = True )
st.sidebar.image('pngwing.com (3).png', caption= 'Welcome User')       


# tenure = st.selectbox('TENURE', data.TENURE.unique())
# montant = st.number_input('MONTANT', data.MONTANT.min(), data.MONTANT.max())
# freq_rech = st.number_input('FREQUENCE_RECH', data.FREQUENCE_RECH.min(), data.FREQUENCE_RECH.max())
# revenue = st.number_input('REVENUE', data.REVENUE.min(), data.REVENUE.max())
# arpu_segment = st.number_input('ARPU_SEGMENT', data.ARPU_SEGMENT.min(), data.ARPU_SEGMENT.max())
# frequence = st.number_input('FREQUENCE', data.FREQUENCE.min(), data.FREQUENCE.max())
# data_volume = st.number_input('DATA_VOLUME', data.DATA_VOLUME.min(), data.DATA_VOLUME.max())
# no_net = st.number_input('ON_NET', data.ON_NET.min(), data.ON_NET.max())
# regularity = st.number_input('REGULARITY', data.REGULARITY.min(), data.REGULARITY.max())
# new_tenure = encoder.transform([tenure])
# st.write(new_tenure)

tenure = st.sidebar.selectbox('DURATION AS A CUSTOMER', data.TENURE.unique())
montant = st.sidebar.number_input('AMOUNT RELOADED', df.MONTANT.min(), df.MONTANT.max())
freq_rech = st.sidebar.number_input('RELOADS', df.FREQUENCE_RECH.min(), df.FREQUENCE_RECH.max())
revenue = st.sidebar.number_input('MONTHLY INCOME', df.REVENUE.min(), df.REVENUE.max())
arpu_segment = st.sidebar.number_input('INCOME(90 DAYS)', df.ARPU_SEGMENT.min(), df.ARPU_SEGMENT.max())
frequence = st.sidebar.number_input('INCOME FREQUENCY', df.FREQUENCE.min(), df.FREQUENCE.max())
data_volume = st.sidebar.number_input('ACTIVENESS OF CLIENT(90 DAYS)', df.DATA_VOLUME.min(), df.DATA_VOLUME.max())
no_net = st.sidebar.number_input('CALL DURATION', df.ON_NET.min(), df.ON_NET.max())
regularity = st.sidebar.number_input('REGULARITY', df.REGULARITY.min(), df.REGULARITY.max())

new_tenure = encoder.transform([tenure])

input_var = pd.DataFrame({'TENURE': [new_tenure],
                           'MONTANT': [montant], 
                           'FREQUENCE_RECH': [freq_rech],
                          'REVENUE':[revenue],
                           'ARPU_SEGMENT':[arpu_segment],
                            'FREQUENCE':[frequence],
                             'DATA_VOLUME':[data_volume],
                              'ON_NET':[no_net],
                                'REGULARITY':[regularity]})
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h5 style= 'margin: -30px; color:olive; font:sans serif' >", unsafe_allow_html= True)
st.dataframe(input_var)

predicted = model.predict(input_var)
output = None
if predicted[0] == 0:
    output = 'Not Churn'
else:
    output = 'Churn'
# transformed= encoder.transform([predicted])
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The customer is predicted to {output}')
    st.balloons()