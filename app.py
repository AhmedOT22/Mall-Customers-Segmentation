import streamlit as st
from src.models.prediction import predict
from src.models.storage import load_model, load_csv
from src.config import MODEL_PATH, PROCESSED_DATA_PATH, FEATURES, CLUSTER_INFO
from src.utils.form import get_user_input, load_custom_styles
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Streamlit Setup
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
load_custom_styles()
st.markdown("<h1>Mall Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p>
        This tool helps segment mall customers based on their demographics and shopping behavior using machine learning.
        By identifying customer personas, marketers can personalize campaigns, improve product targeting, and enhance shopping experiences.
        <br><br>
        <strong>How it works:</strong> Enter the customer’s age, income, and spending score to classify them into a persona cluster.
        Our model analyzes this input and matches it to behavioral groups previously discovered in the dataset.
    </p>
    """,
    unsafe_allow_html=True
)
# Load resources
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

@st.cache_data
def load_clustered_data():
    return load_csv(PROCESSED_DATA_PATH)

model = load_trained_model()
df = load_clustered_data()

# Get user input
with st.container():
    form_col1, form_col2, form_col3 = st.columns([1, 2, 1])
    with form_col2:
        with st.form(key="input_form"):
            st.markdown("""
            <p style='margin-bottom:1.5rem; color:#555; font-size: 0.95rem;'>
                <strong>Fill in the fields below to predict a customer persona:</strong><br>
            </p>
            """, unsafe_allow_html=True)

            age, income, score = get_user_input()
            input_dict = {'Age': age, 'Annual_Income': income, 'Spending_Score': score}
            submit = st.form_submit_button("Predict Customer Persona")

# Prediction
if submit and model:
    try:
        cluster = predict(model, input_dict, FEATURES)
        cluster_data = CLUSTER_INFO.get(cluster, {
            "name": f"Cluster {cluster}",
            "description": "No information available.",
            "recommendation": "No recommendation available."
        })

        st.markdown("<hr style='margin-top: 2rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="result-grid">
                <div class="grid-row card-row-1">
                    <div class="label-box">Persona</div>
                    <div class="content-box">{cluster_data['name']}</div>
                </div>
                <div class="grid-row card-row-2">
                    <div class="label-box">Description</div>
                    <div class="content-box">{cluster_data['description']}</div>
                </div>
                <div class="grid-row card-row-3">
                    <div class="label-box">Recommended Brands</div>
                    <div class="content-box">{cluster_data['recommendation']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.error(f"Prediction error: {e}")
