import streamlit as st

def get_user_input():
    """
    Collects customer attributes from the sidebar input form.

    Returns:
        submit (bool): Whether the user clicked the predict button.
        input_dict (dict): Dictionary containing Age, Annual Income, and Spending Score.
    """
    with st.container():
        st.subheader("Enter Customer Information")

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

        submit = st.button("Predict Cluster")

    return submit, {
        'Age': age,
        'Annual_Income': income,
        'Spending_Score': score
    }
