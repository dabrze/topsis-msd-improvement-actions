from MSDTransformer import MSDTransformer
import numpy as np
import pandas as pd
import pytest

def test_checkInput_weights_length(df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info:
        buses.fit(df, [1], objectives, None)
    assert err_info.type is ValueError

def test_checkInput_weights_value(df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError)  as err_info:
        buses.fit(df, [1, 1, 1, 1, 'test', 1, 1, 1], objectives, None)
    assert err_info.type is ValueError

def test_checkInput_objectives_length(df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info :
        buses.fit(df, None, objectives, None)
    assert err_info.type is ValueError

def test_checkInput_objectives_value(df):
    objectives = ['max', 'max', 'min', 'test', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError)  as err_info:
        buses.fit(df, None, objectives, None)
    assert err_info.type is ValueError

def test_checkInput_expert_range_length(df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info :
        buses.fit(df, None, objectives, [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    assert err_info.type is ValueError

def test_checkInput_expert_range_values_number(df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info :
        buses.fit(df, None, objectives, [[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    assert err_info.type is ValueError

def test_checkInput_expert_range_numerical_value(df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info :
        buses.fit(df, None, objectives, [['a', 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    assert err_info.type is ValueError

def test_checkInput_expert_range_min_max(df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info :
        buses.fit(df, None, objectives, [[5, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    assert err_info.type is ValueError
