from MSDTransformer import MSDTransformer
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
    result = [22.000, -18.875, 0.000]
    assert buses.improvement['Change'].values.tolist() == result
