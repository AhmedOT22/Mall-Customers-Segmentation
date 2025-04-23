import streamlit as st

def get_user_input():
    """
    Collects customer attributes for prediction.

    Returns:
        tuple: age, income, score (as individual values)
    """
    age = st.slider(
        "Age",
        min_value=18,
        max_value=70,
        value=30,
        help="Approximate age of the customer."
    )

    income = st.slider(
        "Annual Income (in $1000s)",
        min_value=15,
        max_value=150,
        value=60,
        help="Estimate of annual income before tax."
    )

    score = st.slider(
        "Spending Behavior on Clothing",
        min_value=1,
        max_value=100,
        value=50,
        help="Estimate how much of their income the customer is likely to spend on clothing, from 1 (very little) to 100 (a lot)."
    )

    return age, income, score


def load_custom_styles():
    """
    Loads external CSS styling for the app from src/assets/style.css.
    """
    try:
        with open('src/assets/style.css') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load custom styles: {e}")