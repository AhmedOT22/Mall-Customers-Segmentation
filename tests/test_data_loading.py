import pytest
from data_processing.data_loader import load_data
import pandas as pd

def test_load_data_success():
    df = load_data('src/data/raw/mall_customers.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_data_fail():
    with pytest.raises(FileNotFoundError):
        load_data('src/data/raw/missing_file.csv')
