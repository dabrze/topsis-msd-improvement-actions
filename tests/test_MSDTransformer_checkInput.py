from src.MSDTransformer import MSDTransformer
import numpy as np
import pandas as pd
import pytest

@pytest.mark.parametrize("weights", [([1]), ([1, 1, 1, 1, 'test', 1, 1, 1])])
def test_checkInput_weights(weights, df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info:
        buses.fit(df, weights, objectives, None)
    assert err_info.type is ValueError

@pytest.mark.parametrize("objectives", [(['max', 'max', 'min', 'max', 'min', 'min', 'min']), (['max', 'max', 'min', 'test', 'min', 'min', 'min', 'max'])])
def test_checkInput_objectives(objectives, df):
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info:
        buses.fit(df, None, objectives, None)
    assert err_info.type is ValueError

@pytest.mark.parametrize("expert_range", [([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
                                          ([[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
                                          ([['a', 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
                                          ([[5, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])])
def test_checkInput_expert_range(expert_range, df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info:
        buses.fit(df, None, objectives, expert_range)
    assert err_info.type is ValueError
