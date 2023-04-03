import pytest
import pandas as pd

@pytest.fixture(autouse = True)
def df():
  data = pd.read_csv("src/bus.csv", sep = ';', index_col = 0)
  return data
