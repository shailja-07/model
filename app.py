
import joblib  
import streamlit as st





model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')


def predict_heart_disease(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return prediction[0]


st.title("! Heart disease prediction model !")
st.header("Are you prone to heart diseases??", divider=True)
st.subheader("complete the  entries given below to know the answer", divider=True)


age = st.number_input("Age", min_value=25, max_value=80, value=30)
st.divider()


sex = st.radio("gender", [0, 1],captions=["female","male"])  
st.divider()

cp = st.radio("Chest Pain Type", [0, 1, 2, 3],captions=["Typical angina","Atypical angina","Non-angina pain","Asymptomatic"])  
st.divider()
    
trestbps = st.number_input("Resting Blood Pressure", min_value=90,max_value=200, value=120)
st.divider()

chol = st.number_input("Cholesterol", min_value=120,max_value=580, value=200)
st.divider()

fbs = st.radio("Fasting Blood Sugar", [0, 1],captions=["< 120 mg/dL","> 120 mg/dL"])
st.divider()

restecg = st.radio("Resting Electrocardiographic Results", [0, 1, 2],captions=["Normal","Abnormal ST-T wave","Showing probable or definite left ventricular hypertrophy"])
st.divider()

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70,max_value=210, value=150)
st.divider()
exang = st.radio("Exercise Induced Angina", [0, 1],captions=["Negative","Positive"])
st.divider()
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, value=0.0)
st.divider()
slope = st.radio("Slope of the Peak Exercise ST Segment", [0, 1, 2],captions=["Upsloping","Horizontal","Downsloping"])
st.divider()

ca = st.number_input("Number of Major Vessels coloured by fluoroscopy  ", min_value=0, max_value=3, value=0)
st.divider()
thal = st.number_input("Thallium scintigraphy",min_value=0,max_value=7) 
st.divider()

input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]


if st.button("Predict"):
    prediction = predict_heart_disease(input_data)
    if prediction == 1:
        st.success("you are prone to heart diseases.")
    else:
        st.success("your are healthy.")


st.feedback("stars")