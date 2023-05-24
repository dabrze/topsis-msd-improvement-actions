from src.MSDTransformer import MSDTransformer
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
    buses = MSDTransformer()
    buses.fit(df, None, objectives, None)
    buses.transform()
    buses.improvement_features(27,3,0.01, ['MaxSpeed', 'Blacking', 'SummerCons'])
    for alt_name in buses.improvement.index:
        if objectives[alt_name] == 'max':
            assert buses.improvement.loc[alt_name]['Change'] >= 0
        elif objectives[alt_name] == 'min':
            assert buses.improvement.loc[alt_name]['Change'] <= 0
            
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
    buses = MSDTransformer()
    buses.fit(df, None, objectives, None)
    buses.transform()
    buses.improvement_features(27,3,0.01, ['MaxSpeed', 'Blacking', 'SummerCons'])
    result = np.array([22.000, -10.25, 0.000])
    assert buses.improvement['Change'].to_numpy() == pytest.approx(result, abs=1e-3)
