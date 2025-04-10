import pickle
from src.models.prediction import predict
from src.config import FEATURES

def test_predict_cluster():
    with open('models/kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)

    input_dict = {'Age': 25, 'Annual_Income': 60, 'Spending_Score': 40}
    cluster = predict(model, input_dict, FEATURES)

    assert isinstance(cluster, int)
    assert cluster >= 0
