import streamlit as st
from recommendation_generator import get_doc

st.title("Doctor Recommendation Tool 🥼")

question = st.text_input("Enter your symptoms: ")

if question:
    response = get_doc(question)

    st.header("Recommendation: ")
    st.write(response)

