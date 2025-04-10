# Mall Customer Segmentation App

Streamlit application that segments mall customers based on their age, annual income, and clothing-related spending behavior. It uses unsupervised clustering (KMeans) to predict the customer's persona and recommend clothing brands accordingly.

![Scikit-learn](https://img.shields.io/badge/framework-scikit--learn-blue)
![Streamlit](https://img.shields.io/badge/ui-streamlit-orange)
![Clustering](https://img.shields.io/badge/model-KMeans-lightgrey)


## Live Demo

*Coming Soon*

## Features

- Loads and processes customer demographic and behavioral data
- Trains and evaluates KMeans clustering models using WSS and silhouette metrics
- Saves the trained model and clustered data for inference
- Provides a user-friendly form for simulating new customers
- Predicts which customer persona the user belongs to
- Displays cluster profile, recommended clothing brands, and 2D cluster visualization
- Includes centralized logging and configuration
- Modularized and testable code structure (enterprise-ready)


## Dataset
- Path: `data/raw/mall_customers.csv`
- Features Used:
  - Numerical: `Age`, `Annual_Income` (in $1000s), `Spending_Score` (proxy for clothing spend)


## Model Architecture

- **Algorithm**: KMeans Clustering
- **Evaluation**: WSS (Elbow Method), Silhouette Score
- **Cluster Count**: Selected via optimal score (`k=6`)
- **Post-Clustering**: Each cluster is mapped to a descriptive persona with recommended clothing brands


## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/mall-segmentation-app.git
   cd mall-segmentation-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Launch the app**
   ```bash
   streamlit run app.py
   ```

   
## Results
- Optimal number of clusters: **6**
- Persona examples:
  - `Affluent Spender`: high income and spending → Hugo Boss, Calvin Klein
  - `Stretch Shopper`: low income, high spending → H&M, Gap
  - `Cautious Minimalist`: low income, low spending → Walmart Apparel, Joe Fresh
- Interactive 2D cluster plot based on Annual Income × Spending Score


## Requirements
- Python ≥ 3.8
- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn


## Author
Developed by [Ahmed Ouazzani](https://github.com/AhmedOT22)


## License
MIT License © 2025