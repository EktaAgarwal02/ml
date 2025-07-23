import streamlit as st
import joblib
import numpy as np

#load the model
with open('models/xgboost.pkl','rb') as file:
    model = joblib.load(file)
    #title
st.set_page_config(page_title= "Outcome Predictor")
st.title("Outcome Predictor")
st.image("https://www.suratdiabeticfootcare.in/blog/image/Importanc-of-Diabetics.webp", width=400)
st.subheader("Predict your outcome based on features")
#st.markdown("abcd")#css
#st.text("abc") 
#sidebar
st.sidebar.header("enter your details👍")
# Pregnancies
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
Pregnancies
</div>
""", unsafe_allow_html=True)
Pregnancies = st.sidebar.slider("", min_value=0, max_value=20, step=1)

# Glucose
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
Glucose
</div>
""", unsafe_allow_html=True)
Glucose = st.sidebar.slider("", min_value=0, max_value=200, step=1)

# Blood Pressure
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
Blood Pressure
</div>
""", unsafe_allow_html=True)
BloodPressure = st.sidebar.slider("", min_value=0, max_value=140, step=1)

# Skin Thickness
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
Skin Thickness
</div>
""", unsafe_allow_html=True)
SkinThickness = st.sidebar.slider("", min_value=0, max_value=100, step=1)

# Insulin
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
Insulin
</div>
""", unsafe_allow_html=True)
Insulin = st.sidebar.slider("", min_value=0, max_value=900, step=10)

# BMI
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
BMI
</div>
""", unsafe_allow_html=True)
BMI = st.sidebar.slider("", min_value=0.0, max_value=70.0, step=0.1)

# Diabetes Pedigree Function
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
Diabetes Pedigree Function
</div>
""", unsafe_allow_html=True)
DiabetesPedigreeFunction = st.sidebar.slider("", min_value=0.0, max_value=3.0, step=0.01)

# Age
st.sidebar.markdown("""
<div style='background-color:black; padding:6px; border-radius:5px; color:white; text-align:center;'>
Age
</div>
""", unsafe_allow_html=True)
Age = st.sidebar.slider("", min_value=1, max_value=100, step=1)

if st.sidebar.button("Predict Outcome"):
    #predict outcome
    outcome = model.predict(np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))[0]
    #display result
    st.markdown(
    f"""
    <div style='background-color: white; padding: 15px; border-radius: 10px; 
                color: green; font-weight: bold; font-size: 16px;margin-bottom: 20px;'>
        Predicted Outcome (0 = Non-Diabetic, 1 = Diabetic): {outcome:.2f}
    </div>
    """,
    unsafe_allow_html=True
)


        #additionl info
    st.markdown(
    f"""
    <div style='background-color: white; padding: 15px; border-radius: 10px; 
                color: green; font-weight: bold; font-size: 16px;'>
        this predict is based on a XGBoost model
    </div>
    """,
    unsafe_allow_html=True
)

#footer
st.markdown(
    """
    <style>
    .stApp {
    background-color: #f5f5f5;
    background-image:linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("https://www.suratdiabeticfootcare.in/blog/image/Importanc-of-Diabetics.webp");
    background-size: cover;
    font-family: 'Arial';
    color: white;
}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-image: 
            linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
            url("https://www.careinsurance.com/upload_master/media/posts/March2020/aysq8NKHywtTzITjLDNC.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: white;
        font-weight: bold;
    }

    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown('-------')
st.markdown("made with using streamlit")