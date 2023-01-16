from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import itertools
import plotly.graph_objects as go

import time


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


    def plot(self):

        #tic = time.perf_counter()

        ### for all possible mean and std count aggregation value and color it by it

        precision = 500
        tempx = [] #Mean
        tempy = [] #Std
        tempc = [] #AggFn

        for x in range(0,precision):
            for y in range(0,int(precision/2)):
                tempx.append(x/precision)
                tempy.append(y/precision)

        if type(self.agg_fn) == str:
            if self.agg_fn == 'I':
                for i in range(len(tempx)):
                    tempc.append(1 - np.sqrt((1-tempx[i])*(1-tempx[i])+(tempy[i]*tempy[i])))
            elif self.agg_fn == 'A':
                for i in range(len(tempx)):
                    tempc.append(np.sqrt(tempx[i]*tempx[i]+(tempy[i]*tempy[i])))
            elif self.agg_fn == 'R':
                for i in range(len(tempx)):
                    tempc.append((np.sqrt(tempx[i]*tempx[i]+(tempy[i]*tempy[i])))/(((1 - np.sqrt((1-tempx[i])*(
                        1-tempx[i])+(tempy[i]*tempy[i])))-1)*(-1) + (np.sqrt(tempx[i]*tempx[i]+(tempy[i]*tempy[i])))))
        else:
            for i in range(len(tempx)):
                tempc.append(self.agg_fn)
        

        fig = go.Figure(data = go.Contour(
                    x=tempx,
                    y=tempy,
                    z=tempc,
                    zmin=0.0,
                    zmax=1.0,
                    colorscale = 'jet',
                    contours_coloring='heatmap',
                    line_width = 0,
                    colorbar = dict(
                        title='Aggregation value',
                        titleside='right',
                        outlinewidth=1,
                        title_font_size=22,
                        tickfont_size=15

                        
                    ),
                    hoverinfo='none'),
            layout=go.Layout(
                title=go.layout.Title(
                    text="Visualizations of dataset in MSD-space",
                    font_size=30
                    ),
                title_x = 0.5,
                xaxis_range = [0.0, 1.0],
                yaxis_range = [0.0, 0.5]
            )
            )
        
        fig.update_xaxes(
                        title_text="M: mean",
                        title_font_size=22,
                        tickfont_size=15,
                        tickmode='auto',
                        showline=True, 
                        linewidth=1.25, 
                        linecolor='black',
                        minor=dict(
                            ticklen=6,
                            ticks="inside", 
                            tickcolor="black", 
                            showgrid=True
                            )
                        )
        fig.update_yaxes(
                        title_text="SD: std",
                        title_font_size=22,
                        tickfont_size=15,
                        showline=True, 
                        linewidth=1.25, 
                        linecolor='black',
                        minor=dict(
                            ticklen=6,
                            ticks="inside",
                            tickcolor="black", 
                            showgrid=True
                            )
                        )


        ### for all values 0.0-1.0 create all possible combinations and for each count mean and std values

        precision = 6
        tempset = []

        for i in range(precision+1):
            tempset.append(round(i/precision, 2))

        temp_DataFrame1 = pd.DataFrame(list(itertools.product(tempset, repeat=int(len(self.data.columns.values[:-3])))), columns=self.data.columns.values[:-3])

        temp_DataFrame1['Mean']=round(temp_DataFrame1.mean(axis=1),2)
        temp_DataFrame1['Std']=round(temp_DataFrame1.std(axis=1),3)

        temp_DataFrame1.sort_values(by=['Mean','Std'])

        temp_DataFrame = pd.DataFrame(temp_DataFrame1.groupby(['Mean'])['Std'].max().reset_index())


        ### distance between before and after

        j = 1
        treshole = 0.015
        indexes = temp_DataFrame.index.tolist()
        isGood = False
        
        for k in range(1,len(temp_DataFrame)-1):
            j=k
            if(not isGood):

                isGood = True

                for i in range(k,len(temp_DataFrame)-1):
        
                    Dif_B = temp_DataFrame['Std'][indexes[j]] - temp_DataFrame['Std'][indexes[j-1]]
                    Dif_A = temp_DataFrame['Std'][indexes[j]] - temp_DataFrame['Std'][indexes[j+1]]
                    

                    if ((Dif_A>=0 or Dif_B>=0) or ((Dif_B >= (-1)*treshole and Dif_A >= (-1)*treshole))):
                        j+=1
                    else:
                        isGood = False
                        temp_DataFrame.drop(indexes[j], inplace = True)
                        temp_DataFrame.reindex(list(range(0,len(temp_DataFrame))))
                        indexes = temp_DataFrame.index.tolist()



        if type(self.agg_fn) == str:
            if self.agg_fn == 'I':
                temp_DataFrame['AggFn'] = 1 - np.sqrt((1-temp_DataFrame['Mean'])*(
                    1-temp_DataFrame['Mean'])+(temp_DataFrame['Std']*temp_DataFrame['Std']))
                topsis_val = temp_DataFrame['AggFn']
            elif self.agg_fn == 'A':
                temp_DataFrame['AggFn'] = np.sqrt(
                    temp_DataFrame['Mean']*temp_DataFrame['Mean']+(temp_DataFrame['Std']*temp_DataFrame['Std']))
                topsis_val = temp_DataFrame['AggFn']
            elif self.agg_fn == 'R':
                temp_DataFrame['AggFn'] = (np.sqrt(temp_DataFrame['Mean']*temp_DataFrame['Mean']+(temp_DataFrame['Std']*temp_DataFrame['Std'])))/(((1 - np.sqrt((1-temp_DataFrame['Mean'])*(
                    1-temp_DataFrame['Mean'])+(temp_DataFrame['Std']*temp_DataFrame['Std'])))-1)*(-1) + (np.sqrt(temp_DataFrame['Mean']*temp_DataFrame['Mean']+(temp_DataFrame['Std']*temp_DataFrame['Std']))))
                topsis_val = temp_DataFrame['AggFn']
        else:
            temp_DataFrame['AggFn'] = self.agg_fn
            topsis_val = self.agg_fn



        fig.add_trace(go.Scatter(
            x=temp_DataFrame['Mean'],
            y=temp_DataFrame['Std'],
            mode='lines',
            showlegend = False,
            hoverinfo='none',
            line_color='black'
        ))

        fig.add_trace(go.Scatter(
            x=[0,1],
            y=[0.5,0.5],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 1)',
            showlegend = False,
            hoverinfo='none',
            line_color='white'
        ))
     

        ### plot the ranked data

        custom = []
        for i in self.data.index.values:
            custom.append(1+ self.ranked_alternatives.index(i))


        fig.add_trace(go.Scatter(
            x=self.data['Mean'].tolist(),
            y=self.data['Std'].tolist(),
            showlegend = False,
            mode='markers',
            marker=dict(
                color='black',
                size=10
            )
            ,customdata=custom,
            text=self.data.index.values,
            hovertemplate= '<b>%{text}</b><br>Rank: %{customdata:f}<extra></extra>'
        ))

        fig.show()

        #toc = time.perf_counter()
        #print("Created plot in ", (toc - tic), " seconds")


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

        if(not all(type(item) in [int, float, np.float64] for item in self.weights)):
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
                    raise ValueError("Invalid value at 'expert_range'. Minimal value  is bigger then maximal value.")

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
