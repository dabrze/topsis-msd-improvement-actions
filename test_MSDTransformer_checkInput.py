import pytest
import numpy as np
import pandas as pd
from main_3 import MSDTransformer


def test_checkInput_weights_length():
    
    df = pd.read_csv("bus.csv", sep=';', index_col=0)
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError) as err_info:
        buses.fit(df, [1], objectives, None)
    assert err_info.type is ValueError



def test_checkInput_weights_value():
    
    df = pd.read_csv("bus.csv", sep=';', index_col=0)
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    buses = MSDTransformer('I')
    with pytest.raises(ValueError)  as err_info:
        buses.fit(df, [1, 1, 1, 1, 'test', 1, 1, 1], objectives, None)
    assert err_info.type is ValueError


def test_checkInput_objectives_length():
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min']
        buses = MSDTransformer('I')
        with pytest.raises(ValueError) as err_info :
            buses.fit(df, None, objectives, None)
        assert err_info.type is ValueError


def test_checkInput_objectives_value():
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'test', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        with pytest.raises(ValueError)  as err_info:
            buses.fit(df, None, objectives, None)
        assert err_info.type is ValueError


def test_checkInput_expert_range_length():
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        with pytest.raises(ValueError) as err_info :
            buses.fit(df, None, objectives, [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        assert err_info.type is ValueError


def test_checkInput_expert_range_values_number():
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        with pytest.raises(ValueError) as err_info :
            buses.fit(df, None, objectives, [[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        assert err_info.type is ValueError


def test_checkInput_expert_range_numerical_value():
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        with pytest.raises(ValueError) as err_info :
            buses.fit(df, None, objectives, [['a', 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        assert err_info.type is ValueError


def test_checkInput_expert_range_min_max():
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        with pytest.raises(ValueError) as err_info :
            buses.fit(df, None, objectives, [[5, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        assert err_info.type is ValueError
