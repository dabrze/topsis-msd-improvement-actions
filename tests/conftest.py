import pytest
import pandas as pd


@pytest.fixture(autouse=True)
def df():
    data = pd.read_csv("data/bus_original.csv", sep=";", index_col=0)
    return data
