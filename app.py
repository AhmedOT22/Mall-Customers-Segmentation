import streamlit as st
import pandas as pd
import logging
from src.models.prediction import predict
from src.models.storage import load_model, load_csv
from config import MODEL_PATH, PROCESSED_DATA_PATH, FEATURES, CLUSTER_INFO
from src.utils.form import get_user_input
from src.utils.visualization import plot_scatter_clusters
from src.utils.logger import get_logger
from streamlit_extras.metric_cards import style_metric_cards

logger = get_logger(__name__)

# Streamlit Setup
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("Mall Customer Segmentation")
st.markdown("Predict the type of customer based on their age, income, and spending habits.")

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
with st.sidebar:
    st.header("Customer Input Form")
    st.markdown("Use the sliders to simulate a customer's profile.")
    submit, input_dict = get_user_input()

# Prediction
if submit and model:
    try:
        cluster = predict(model, input_dict, FEATURES)
        cluster_data = CLUSTER_INFO.get(cluster, {
            "name": f"Cluster {cluster}",
            "description": "No information available.",
            "recommendation": "No recommendation available."
        })

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Assigned Cluster", value=f"#{cluster}")
            st.metric(label="Customer Persona", value=cluster_data['name'])

        with col2:
            st.subheader("Customer Profile")
            st.markdown(f"**Description:** {cluster_data['description']}")
            st.markdown(f"**Recommended Brands:** {cluster_data['recommendation']}")

        
        #style_metric_cards()

        #if df is not None:
         #   st.markdown("---")
          #  st.subheader("Visual Cluster Placement (2D)")
#
 #           df_copy = df.copy()
  #          df_copy['User'] = 'Existing'
#
 #           user_df = pd.DataFrame([input_dict])
  #          user_df['Cluster'] = cluster
   #         user_df['User'] = 'You'
#
 #           viz_df = pd.concat([
  #              df_copy[['Annual_Income', 'Spending_Score', 'Cluster', 'User']],
   #             user_df[['Annual_Income', 'Spending_Score', 'Cluster', 'User']]
    #        ])
#
 #           fig = plot_scatter_clusters(viz_df, x='Annual_Income', y='Spending_Score',
  #                                      label_col='Cluster', title='Cluster Visualization')
   #         st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.error(f"Prediction error: {e}")
