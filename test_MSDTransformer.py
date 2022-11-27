import pytest
import numpy as np
import pandas as pd
from main_3 import MSDTransformer


def test_checkInput_weights_length():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        buses.fit(df, [1], objectives, None)
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in weights.")


def test_checkInput_weights_value():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        buses.fit(df, [1, 1, 1, 1, 'test', 1, 1, 1], objectives, None)
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in weights.")


def test_checkInput_objectives_length():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min']
        buses = MSDTransformer('I')
        buses.fit(df, None, objectives, None)
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in objectives.")


def test_checkInput_objectives_value():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'test', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        buses.fit(df, None, objectives, None)
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in objectives.")


def test_checkInput_expert_range_length():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        buses.fit(df, None, objectives, [[0, 1], [0, 1], [
                  0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in expert_range.")


def test_checkInput_expert_range_values_number():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        buses.fit(df, None, objectives, [[1], [0, 1], [0, 1], [
                  0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in expert_range.")


def test_checkInput_expert_range_numerical_value():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        buses.fit(df, None, objectives, [['a', 1], [0, 1], [
                  0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in expert_range.")


def test_checkInput_expert_range_min_max():
    try:
        df = pd.read_csv("bus.csv", sep=';', index_col=0)
        objectives = ['max', 'max', 'min', 'max', 'min', 'min', 'min', 'max']
        buses = MSDTransformer('I')
        buses.fit(df, None, objectives, [[5, 1], [0, 1], [
                  0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    except:
        return
    else:
        raise Exception(
            "CheckInput wrongly doesn't detect invalid value in expert_range.")
