from MSDTransformer import MSDTransformer
import numpy as np
import pandas as pd
import pytest

def test_calculate_copy(df):
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert df.equals(buses.original_data)

def test_calculate_multipleTransform(df):
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    b1 = buses.data
    buses.transform()
    b2 = buses.data
    assert b1.equals(b2)

def test_calculate_ranking(df):
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert buses.ranked_alternatives == ['b24', 'b26', 'b07', 'b16', 'b18', 'b25', 'b04', 'b01', 'b28', 'b09', 'b02', 'b13', 'b11', 'b32', 'b21', 'b12', 'b27', 'b17', 'b06', 'b29', 'b20', 'b14', 'b23', 'b19', 'b03', 'b30', 'b08', 'b22', 'b15', 'b10', 'b31', 'b05']

def test_calculate_normWeigths(df):
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert (buses.weights.all() <= 1 and buses.weights.all() >= 0)

def test_calculate_normData(df):
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer()
    buses.fit(df, objectives=objectives)
    buses.transform()
    assert (buses.data.all().all() <= 1 and buses.data.all().all() >= 0)
    
@pytest.mark.parametrize("col_name, agg_fn, expected", [('Mean', 'I', [0.8471063335888918, 0.7768981460905122, 0.49858239165620366, 0.8586295634766312, 0.18392857142857144]),
                                                        ('Std', 'I', [0.10893897542978036, 0.11494599992993791, 0.22042413986540876, 0.10427543268417871, 0.3289933238575779]), 
                                                        ('AggFn', 'I', [0.8122656834877503, 0.7490278499160838, 0.452271582449278, 0.8243327970727589, 0.12010842504688712]),
                                                        ('AggFn', 'A', [0.8540824554889914, 0.7853555324174957, 0.5451341144205023, 0.8649382019182312, 0.37691686952246145]),
                                                        ('AggFn', 'R', [0.8198010230228603, 0.757825488673488, 0.4988130697803592, 0.8311874970961471, 0.29990001354560974])])
def test_calculate_values(col_name, agg_fn, expected, df):
    objectives = ['max','max','min','max','min','min','min','max']
    buses = MSDTransformer(agg_fn)
    buses.fit(df, objectives=objectives)
    buses.transform()
    result = buses.data[col_name][:5].values.tolist()
    result = [round(num, 6) for num in result]
    result2 = [round(num, 6) for num in expected]
    assert result == result2
