# src/config.py

RAW_DATA_PATH = 'data/raw/mall_customers.csv'
PROCESSED_DATA_PATH = 'data/processed/clustered_customers.csv'
MODEL_PATH = 'models/kmeans_model.pkl'
FIGURE_DIR = 'reports/figures/'

# Clustering configuration
FEATURES = ['Age', 'Annual_Income', 'Spending_Score']
DEFAULT_K = 6
K_RANGE = range(3, 9)


# Can go in src/config.py or as a JSON file
CLUSTER_INFO = {
    0: {
        "name": "Mindful Buyer",
        "description": "Moderately earning and moderately spending. Thinks before purchasing, seeks quality over quantity.",
        "recommendation": "Uniqlo, Old Navy, Gap, Levi’s"
    },
    1: {
        "name": "Affluent Spender",
        "description": "High income and high spending. Enjoys shopping and premium experiences",
        "recommendation": "Hugo Boss, Michael Kors, Fear of God, Ted Baker"
    },
    2: {
        "name": "Stretch Shopper",
        "description": "Low income but high spending. May spend impulsively or use credit.",
        "recommendation": "H&M, Gap, Bershka, Pull&Bear"
    },
    3: {
        "name": "Detached Earner",
        "description": "Solid income but rarely spends. May be minimalistic, skeptical, or shopping-averse.",
        "recommendation": "Muji, Everlane, Columbia, The North Face, Marks & Spencer"
    },
    4: {
        "name": "Balanced Buyer",
        "description": "Middle-to-upper income with strong spending. Active, thoughtful shopper.",
        "recommendation": "Banana Republic, Lululemon, Aritzia"
    },
    5: {
        "name": "Cautious Minimalist",
        "description": "Low income, low spending. Highly frugal, buys only essentials.",
        "recommendation": "Walmart Apparel, Joe Fresh, Primark, No Boundaries"
    }
}
