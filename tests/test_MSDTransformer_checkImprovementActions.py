from src import MSDTransformer as msdt
import numpy as np
import pandas as pd
import pytest

def test_checkImprovedSign(df):
    objectives = {
        "MaxSpeed" : "max",
        "ComprPressure" : "max",
        "Blacking" : "min",
        "Torque" : "max",
        "SummerCons" : "min",
        "WinterCons" : "min",
        "OilCons" : "min",
        "HorsePower" : "max"}
    agg_function = msdt.ITOPSIS
    buses = msdt.MSDTransformer(agg_function)
    buses.fit_transform(df, weights=None, objectives=objectives, expert_range=None)
    changes = buses.improvement(
        'improvement_features', 
        alternative_to_improve= 7,
        alternative_to_overcome= 3,
        features_to_change= ['MaxSpeed', 'Blacking', 'SummerCons'])
    for col_name in changes.columns:
        if objectives[col_name] == 'max':
            assert changes.at[0, col_name] >= 0
        elif objectives[col_name] == 'min':
            assert changes.at[0, col_name] <= 0
            
def test_checkImprovedValue(df):
    objectives = {
        "MaxSpeed" : "max",
        "ComprPressure" : "max",
        "Blacking" : "min",
        "Torque" : "max",
        "SummerCons" : "min",
        "WinterCons" : "min",
        "OilCons" : "min",
        "HorsePower" : "max"}
    agg_function = msdt.ITOPSIS
    buses = msdt.MSDTransformer(agg_function)
    buses.fit_transform(df, weights=None, objectives=objectives, expert_range=None)
    changes = buses.improvement(
        'improvement_features', 
        alternative_to_improve= 7,
        alternative_to_overcome= 3,
        features_to_change= ['MaxSpeed', 'Blacking', 'SummerCons'])
    result = {
        'MaxSpeed': [0.0],
        'ComprPressure': [0.0],
        'Blacking': [-16.946075],
        'Torque': [0.0],
        'SummerCons': [0.0],
        'WinterCons': [0.0],
        'OilCons': [0.0],
        'HorsePower': [0.0]}
    result = pd.DataFrame(result)
    assert np.allclose(result, changes, rtol=1e-5)
