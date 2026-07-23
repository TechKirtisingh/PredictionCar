import streamlit as st
import joblib



model = joblib.load("model.pkl")
vectorizer = joblib.load("vect.pkl")

st.title("Sentiment Analysis App")

st.write("Enter Any Sentence Below. ")

text = st.text_area("Enter Text")

if st.button("Predict"):
    
    text_vector = vectorizer.transform([text])
    
    prediction = model.predict(text_vector)
    
    if prediction[0] == "Positive":

        st.write("😊 Positive")
        st.balloons()

    elif prediction[0] == "Negative":
        st.write("☹️ Negative")
        st.snow()

    else:
        st.write("😐 Neutral")