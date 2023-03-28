from main import MSDTransformer
import numpy as np
import pandas as pd
import pytest

def test_calculate_copy():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert df.equals(buses.data_)

def test_calculate_multipleTransform():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    b1 = buses.data_
    buses.transform()
    b2 = buses.data_
    assert b1.equals(b2)

def test_calculate_ranking():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert buses.ranked_alternatives == ['b24', 'b26', 'b07', 'b16', 'b18', 'b25', 'b04', 'b01', 'b28', 'b09', 'b02', 'b13', 'b11', 'b32', 'b21', 'b12', 'b27', 'b17', 'b06', 'b29', 'b20', 'b14', 'b23', 'b19', 'b03', 'b30', 'b08', 'b22', 'b15', 'b10', 'b31', 'b05']

def test_calculate_normWeigths():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert (buses.weights.all() <= 1 and buses.weights.all() >= 0)

def test_calculate_normData():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert (buses.data.all().all() <= 1 and buses.data.all().all() >= 0)

def test_calculate_mean():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    result = buses.data['Mean'][:5].values.tolist()
    result2 = ['%.5f' % elem for elem in result]
    assert result2 == ['0.84711', '0.77690', '0.49858', '0.85863', '0.18393']

def test_calculate_sd():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    result = buses.data['Std'][:5].values.tolist()
    result2 = ['%.5f' % elem for elem in result]
    assert result2 == ['0.10894', '0.11495', '0.22042', '0.10428', '0.32899']

def test_calculate_topsisI():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    result = buses.data['AggFn'][:5].values.tolist()
    result2 = ['%.5f' % elem for elem in result]
    assert result2 == ['0.81227', '0.74903', '0.45227', '0.82433', '0.12011']

def test_calculate_topsisA():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    result = buses.data['AggFn'][:5].values.tolist()
    result2 = ['%.5f' % elem for elem in result]
    assert result2 == ['0.81227', '0.74903', '0.45227', '0.82433', '0.12011']

def test_calculate_topsisR():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    result = buses.data['AggFn'][:5].values.tolist()
    result2 = ['%.5f' % elem for elem in result]
    assert result2 == ['0.81227', '0.74903', '0.45227', '0.82433', '0.12011']
