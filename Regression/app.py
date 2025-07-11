import streamlit as st
import joblib
import numpy as np

#load the model
with open('models/simple_linear_regression.pkl','rb') as file:
    model = joblib.load(file)

    #title
st.set_page_config(page_title= "Salary Predictor")
st.title("Salary Predict App")
st.subheader("Predict your salary based on your experience")
#st.markdown("abcd")#css
#st.text("abc")    
#sidebar
st.sidebar.header("enter your details👍")
experience = st.sidebar.slider("Year of experience",min_value=0.0,max_value=20.0,step=0.5)
#button to predict 
if st.sidebar.button("Predict Salary"):
    #predict salary
    salary = model.predict(np.array([[experience]]))[0]
    #display result
    st.success(f"Predict Salary:Rs.{salary:,.2f}")

        #additionl info
    st.info("this predict is based on a Simple Linear Regression")

#footer
st.markdown('-------')
st.markdown("made with using streamlit")



