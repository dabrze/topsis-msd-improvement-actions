from main_3 import MSDTransformer
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
    assert (buses.data['Mean'][:5]).values.tolist() == [0.8471063335888918, 0.7768981460905122, 0.49858239165620366, 0.8586295634766312, 0.18392857142857144]

def test_calculate_sd():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert (buses.data['Std'][:5]).values.tolist() == [0.10893897542978036, 0.11494599992993791, 0.22042413986540876, 0.10427543268417871, 0.3289933238575779]

def test_calculate_topsisI():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert (buses.data['Std'][:5]).values.tolist() == [0.8122656834877503, 0.7490278499160838, 0.452271582449278, 0.8243327970727589, 0.12010842504688712]

def test_calculate_topsisA():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform('A')
    assert (buses.data['Std'][:5]).values.tolist() == [0.8540824554889914, 0.7853555324174957, 0.5451341144205023, 0.8649382019182312, 0.37691686952246145]

def test_calculate_topsisR():
    df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert (buses.data['Std'][:5]).values.tolist() == [0.8198010230228603, 0.757825488673488, 0.4988130697803592, 0.8311874970961471, 0.29990001354560974]