from src import WMSDTransformer as wmsdt
import numpy as np
import pandas as pd
import pytest

@pytest.mark.parametrize("weights", [([1]), ([1, 1, 1, 1, 'test', 1, 1, 1])])
def test_checkInput_weights(weights, df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    agg_function = wmsdt.ITOPSIS
    buses = wmsdt.WMSDTransformer(agg_function)
    with pytest.raises(ValueError) as err_info:
        buses.fit_transform(df, weights=weights, objectives=objectives, expert_range=None)
    assert err_info.type is ValueError

@pytest.mark.parametrize("objectives", [(['max', 'max', 'min', 'max', 'min', 'min', 'min']), (['max', 'max', 'min', 'test', 'min', 'min', 'min', 'max'])])
def test_checkInput_objectives(objectives, df):
    agg_function = wmsdt.ITOPSIS
    buses = wmsdt.WMSDTransformer(agg_function)
    with pytest.raises(ValueError) as err_info:
        buses.fit_transform(df, weights=None, objectives=objectives, expert_range=None)
    assert err_info.type is ValueError

@pytest.mark.parametrize("expert_range", [([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
                                          ([[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
                                          ([['a', 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
                                          ([[5, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])])
def test_checkInput_expert_range(expert_range, df):
    objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
    agg_function = wmsdt.ITOPSIS
    buses = wmsdt.WMSDTransformer(agg_function)
    with pytest.raises(ValueError) as err_info:
        buses.fit_transform(df, weights=None, objectives=objectives, expert_range=expert_range)
    assert err_info.type is ValueError
