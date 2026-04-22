import streamlit as st
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9
        ; /* light blue background */
    }
    h1 {
        color: #2e86c1; /* deep blue title */
    }
    .stButton>button {
        background-color: #28a745; /* green button */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("Student Study Hours Prediction")
st.image("student.jpg", width=600, caption="Welcome to Student Study Hours Prediction")


st.sidebar.header("Input Parameters")
gender = st.sidebar.radio("Pick your gender", ["Male", "Female"])
hours = st.sidebar.slider("Choose your study hours", 0, 20, 5)
st.metric("Study Hours", hours)

if st.sidebar.button("Predict"):
    input_data = np.array([[hours]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        result = "You are likely to pass!"
        st.balloons()
        st.subheader(result)
        st.subheader("Great job! Keep up the good work!")
        with st.spinner("Calculating..."):
            time.sleep(1)
    else:
        result = "You might be fail if you don't study more."
        st.warning(result)
        st.subheader("Don't worry! You can improve with more effort!")
        with st.spinner("Calculating..."):
            time.sleep(1)
   

hours_range = np.linspace(0, 10, 100).reshape(-1,1)
probs = model.predict_proba(hours_range)[:,1]

fig, ax = plt.subplots()
ax.plot(hours_range, probs, color="#e74c3c", linewidth=2, label="Decision Boundary")
ax.scatter([hours], [model.predict_proba([[hours]])[0][1]], color="#3498db", s=100,edgecolors="black",label="Your Input")
ax.set_facecolor("#f9f9f9")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Probability of Passing")
ax.legend()
st.pyplot(fig)


feedback = st.text_area("Any feedback about this prediction?")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")



