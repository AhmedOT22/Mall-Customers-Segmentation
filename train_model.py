from src.models.training import run_training_pipeline
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH

if __name__ == "__main__":
    run_training_pipeline(
        raw_data_path= RAW_DATA_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        model_path= MODEL_PATH,
        optimal_k=6
    )
