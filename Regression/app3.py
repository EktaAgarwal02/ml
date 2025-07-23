import streamlit as st
import joblib
import numpy as np

#load data
with open('models/xgboost.pkl','rb') as file:
    model = joblib.load(file)
#title
st.set_page_config(page_title="Outcome Predictor")
st.title("Outcome Predictor")

st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJn7hkdR37yGcsXrwutlrRsK4gob4AE4HI6A&s",width = 400)
st.subheader("Predict your outcome based on features")

st.sidebar.header("Enter the details👍")
Pregnancies = st.sidebar.slider("Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.sidebar.slider(" Glucose",min_value=0,max_value=20,step=1)
BloodPressure = st.sidebar.slider("BloodPressure",min_value=0,max_value=200,step=1)
SkinThickness = st.sidebar.slider("SkinThickness",min_value=0,max_value=400,step=1)
Insulin =  st.sidebar.slider("Insulin",min_value=0,max_value=900,step=1)
BMI = st.sidebar.slider("BMI",min_value=0.0,max_value = 90.0 ,step=1.0)
DiabetesPedigreeFunction= st.sidebar.slider("DiabetesPedigreeFunction",min_value=0,max_value=80,step=1)
Age = st.sidebar.slider("Age",min_value=0,max_value=70,step=1)

if st.sidebar.button("Outcome Predictor"):

    Outcome= model.predict(np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]))[0]
    #display result
    st.success(f"Predict the outcome :{int(Outcome)}")
        #additionl info
    st.info("this predict is based on a xgboost")
#footer
    st.markdown('-------')
    st.markdown("made with using streamlit")


