import streamlit as st
import joblib
model=joblib.load("fake_news_detection_model.pkl")
sentiment_labels={'1':'true','0':'false'}
st.title('Fake vs True')
user_input=st.text_area("Enter your text here")
if st.button("predict"):
    print(user_input)
    predicted_sentiment=model.predict([user_input])[0]
    predicted_sentiment_label=sentiment_labels[str(predicted_sentiment)]

    st.info(f"predicted news: {predicted_sentiment_label}")
