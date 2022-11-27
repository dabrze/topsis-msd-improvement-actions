from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class MSDTransformer(TransformerMixin):
    """
    A class to calculate and show TOPSIS ranking of provided dataset

    ...

    Attributes
    ----------
    data : dataframe
        dataframe with data we are working on with few columns added during computations
    x : int
        number of columns
    y : int
        number of rows
    weights : np.array
        array of weights for criterias
    objectives : np.array
        array of types of criterias (gain or cost)
    expert_range
        range of values for criterias given by expert
    isFitted : bool
        a flag to tell if data is fittet

    Methods
    -------
    fit()
        fits the data to make it easier to work on it
    transform()
        performes any nesesary operation to prepare the ranking
    inverse_transform()
        returns the data to the first form
    normalizeData(data)
        normalize given data using either given expert range or min/max
    normalizeWeights(weights)
        normalize weights to make all of them at most 1
    calulateMean()
        calculates and ads mean column to dataframe
    calulateSD()
        calculates and ads standard dewiation column to dataframe
    topsis()
        calculates and ads topsis value column to dataframe
    ranking()
        creates a ranking from the data based on topsis value column
    """

    # ---------------------------------------------------------
    # EXTERNAL FUNCTIONS
    # ---------------------------------------------------------

    def __init__(self, agg_fn='I'):
        """
        Parameters
        ----------
        data : dataframe
            dataframe on whih the algorythm will be performed
        weights : np.array, optional
            array of length equal to number of critetrias (defoult: ones)
        objectives : optional
            array of length equal to number of criterias or a single string (to aply to every criteria), possible values: 'min', 'max', 'gain', 'cost', 'g', 'c'
            (defoult: array of 'max')
        expert_range : np.array, optional
            array of length equal to number of critetrias with minimal and maximal value for every criterion (defoult: none)
        agg_fn : string, optional
            aggregation function to be used to calculate the TOPSIS value (defoult: 'I')
        """

        # [optional] I R A or custom function: default I
        self.agg_fn = (agg_fn if type(agg_fn) == str else agg_fn)

        # flag if the data is not fitted
        self.isFitted = False

    def fit(self, data, weights=None, objectives=None, expert_range=None):
        """fits the data to make it easier to work on it.

        normalises data and weights
        """

        # data
        self.data = data

        # store information about number of rows and number of columns (excluding headers)
        # number of columns (-1, because we must exclude 0th column - alternatives)
        self.m = self.data.shape[1]
        # number of rows    (-1, because we must exclude 0th row - criteria)
        self.n = self.data.shape[0]

        # [optional] criteria weights: default 1, 1, ... 1
        self.weights = (weights if weights is not None else np.ones(self.m))

        # [optional] which criteria should be min, which max: deault max, max, ... max
        # allowed values: 'min', 'max', 'gain', 'cost', 'g', 'c'
        # allowed types: single string or list of strings (len must be equal self.m)
        if(type(objectives) == list):
            self.objectives = objectives
        # when user will only give one value, it will be copied for all criteria
        elif(type(objectives) == str):
            self.objectives = np.repeat(objectives, self.m)
        elif(type(objectives) == None):
            self.objectives = np.repeat('max', self.m)

        # replace all "gain" ang "g" given by the user for 'max'
        self.objectives = list(
            map(lambda x: x.replace('gain', 'max'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('g', 'max'), self.objectives))

        # replace all "cost" ang "c" given by the user for 'min'
        self.objectives = list(
            map(lambda x: x.replace('cost', 'min'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('c', 'min'), self.objectives))

        # [optional] expert range: default None
        self.expert_range = expert_range

        # store values of caluclating ranked alternatives, mean, sd and topsis value
        self.mean_col = []
        self.sd_col = []
        self.topsis_val = []
        self.ranked_alternatives = []

        self.checkInput()

        # create a copy of data to avoid changes to original dataset
        data_ = self.data.copy()
        weights_ = self.weights.copy()

        # normalize data (expert range and objectives)
        self.data = self.normalizeData(data_)

        # normalize weights
        self.weights = self.normalizeWeights(weights_)

        self.isFitted = True

        return self.data

    def transform(self):
        """performes any nesesary operation to prepare the ranking

        calculates and adds mean, tsandard dewiation and topsis value columns to the dataframe and ranks the data

        Rises
        -----
        Exception
            if the data is not fitted
        """

        if(not self.isFitted):
            raise Exception("fit is required before transform")

        # MSD transformation
        self.mean_col = self.calulateMean()
        self.sd_col = self.calculateSD()
        self.topsis_val = self.topsis()

        # ranking
        self.ranked_alternatives = self.ranking()

        return self

    def inverse_transform(self, target):
        # TO DO
        target_ = target.copy()

        return target_

    def plot(self, target):
        # TO DO
        x = np.array(self.data['Mean'])
        y = np.array(self.data['Std'])
        colors = np.array([self.data['AggFn']])

        plt.scatter(x, y, c=colors, cmap='jet')
        for i, txt in enumerate(self.ranked_alternatives):
            plt.annotate(txt, (x[i], y[i]))
        plt.ylim(0, 0.5)
        plt.xlim(0, 1)
        plt.title("Visualizations of dataset in MSD-space")
        plt.xlabel("M: mean(u)")
        plt.ylabel("SD: std(u)")
        plt.colorbar(label="Aggregation value")
        plt.show()

        return

    # ---------------------------------------------------------
        # INTERNAL FUNCTIONS
    # ---------------------------------------------------------

    # def setObjectives(self, passed_objectives):

        # if(passed_objectives == None):
        #    self.objectives = np.repeat( 'max', self.n)
        # elif(type(passed_objectives) == str): #when user will only give one value, it will be copied for all criteria
        #    self.objectives == np.repeat( passed_objectives, self.n)
        # elif(type(passed_objectives) == list):
        #    self.objectives == passed_objectives

        # replace all "gain" ang "g" given by the user for 'max'
        #self.objectives = list(map(lambda x: x.replace('gain', 'max'), self.objectives))
        #self.objectives = list(map(lambda x: x.replace('g', 'max'), self.objectives))

        # replace all "cost" ang "c" given by the user for 'min'
        #self.objectives = list(map(lambda x: x.replace('cost', 'min'), self.objectives))
        #self.objectives = list(map(lambda x: x.replace('c', 'min'), self.objectives))

    def checkInput(self):
        if (len(self.weights) != self.m):
            raise ValueError("Invalid value 'weights'.")

        if(not all(type(item) in [int, float] for item in self.weights)):
            raise ValueError("Invalid value 'weights'. Expected numerical value (int or float).")

        if (len(self.objectives) != self.m):
            raise ValueError("Invalid value 'objectives'.")

        if(not all(item in ["min", "max"] for item in self.objectives)):
            raise ValueError(
                "Invalid value at 'objectives'. Use 'min', 'max', 'gain', 'cost', 'g' or 'c'.")

        if(self.expert_range != None):
            if(len(self.expert_range) != len(self.objectives)):
                raise ValueError(
                    "Invalid value at 'expert_range'. Length of should be equal to number of critetrias.")
            for col in self.expert_range:
                if(len(col) != 2):
                    raise ValueError(
                        "Invalid value at 'expert_range'. Every criterion has to have minimal and maximal value.")
                if(not all(type(item) in [int, float] for item in col)):
                    raise ValueError(
                        "Invalid value at 'expert_range'. Expected numerical value (int or float).")
                if(col[0] > col[1]):
                    raise ValueError("Invalid value at 'expert_range'. Minimal value " +
                                     col[0]+" is bigger then maximal value "+col[1]+".")

    def normalizeData(self, data):
        """normalize given data using either given expert range or min/max

        uses the minmax normalization with minimum and maximum taken from expert ranges if given

        Parameters
        ----------
        data : dataframe
            data to be normalised
        """
        # TO DO
        if self.expert_range is None:
            data = (data-data.min())/(data.max()-data.min())
        else:
            c = 0
            for col in data.columns:
                data[col] = (data[col] - self.expert_range[c][0]) / \
                    (self.expert_range[c][1]-self.expert_range[c][0])
                c += 1
        for i in range(self.m):
            if self.objectives[i] == 'min':
                data[data.columns[i]] = 1 - data[data.columns[i]]
        return data

    def normalizeWeights(self, weights):
        """normalize weights

        result are weights not greater than 1 but not 0 if not present previously

        Parameters
        ----------
        weights : np.array
            weights to be normalised
        """
        # TO DO
        weights = np.array([float(i)/max(weights) for i in weights])
        return weights

    def calulateMean(self):
        """calculates and ads mean column to dataframe"""
        # TO DO
        self.data['Mean'] = self.data.mean(axis=1)
        return self.data['Mean']

    def calculateSD(self):
        """calculates and ads standard dewiatiom column to dataframe"""
        # TO DO
        self.data['Std'] = self.data.std(axis=1)
        return self.data['Std']

    def topsis(self):
        """calculates and ads topsis value column to dataframe"""
        # TO DO
        if type(self.agg_fn) == str:
            if self.agg_fn == 'I':
                self.data['AggFn'] = 1 - np.sqrt((1-self.data['Mean'])*(
                    1-self.data['Mean'])+(self.data['Std']*self.data['Std']))
                topsis_val = self.data['AggFn']
            elif self.agg_fn == 'A':
                self.data['AggFn'] = np.sqrt(
                    self.data['Mean']*self.data['Mean']+(self.data['Std']*self.data['Std']))
                topsis_val = self.data['AggFn']
            elif self.agg_fn == 'R':
                self.data['AggFn'] = (np.sqrt(self.data['Mean']*self.data['Mean']+(self.data['Std']*self.data['Std'])))/(((1 - np.sqrt((1-self.data['Mean'])*(
                    1-self.data['Mean'])+(self.data['Std']*self.data['Std'])))-1)*(-1) + (np.sqrt(self.data['Mean']*self.data['Mean']+(self.data['Std']*self.data['Std']))))
                topsis_val = self.data['AggFn']
        else:
            self.data['AggFn'] = self.agg_fn
            topsis_val = self.agg_fn
        return topsis_val

    def ranking(self):
        """creates a ranking from the data based on topsis value column"""
        # TO DO
        data__ = self.data.copy()
        data__ = data__.sort_values(by='AggFn', ascending=False)
        arranged = data__.index.tolist()
        return arranged
        ###
        #arranged = self.alternatives.copy()
        #val = self.topsis_val.argsort()
        #arranged = arranged[val[::-1]]
        # return arranged
        ###
        #arranged = []
        #arranged = self.data.copy()
        #arranged['R'] = self.data.topsis_val['R']
        #arranged = arranged.sort('R', ascending = False)
        # return arranged[:-1]
