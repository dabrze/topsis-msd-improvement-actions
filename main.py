from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
from IPython.display import display


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
        agg_fn : string or own function, optional
            aggregation function to be used to calculate the TOPSIS value (defoult: 'I')
        """

        self.agg_fn = (agg_fn if type(agg_fn) == str else agg_fn)
        self.isFitted = False

    def fit(self, data, weights=None, objectives=None, expert_range=None):
        """fits the data to make it easier to work on it.

        Parameters
        ----------
        data : dataframe
            dataframe on whih the algorythm will be performed
        weights : np.array, optional
            array of length equal to number of critetrias (default: np.ones())
        objectives : optional
            array of length equal to number of criterias or a single string (to aply to every criteria), possible values: 'min', 'max', 'gain', 'cost', 'g', 'c'
            (default: array of 'max')
        expert_range : np.array, optional
            array of length equal to number of critetrias with minimal and maximal value for every criterion (defoult: none)

        normalises data and weights
        """

        self.original_data = data

        self.data = self.original_data.copy()
        self.m = self.data.shape[1]
        self.n = self.data.shape[0]

        self.original_weights = (weights if weights is not None else np.ones(self.m))
        self.weights = self.original_weights.copy()

        self.objectives = objectives

        if(type(objectives) is list):
            self.objectives = objectives
        elif(type(objectives) is str):
            self.objectives = np.repeat(objectives, self.m)
        elif(type(objectives) is dict):
            self.objectives = self.dictToObjectivesList(objectives)
        elif(objectives is None):
            self.objectives = np.repeat('max', self.m)

        self.objectives = list(
            map(lambda x: x.replace('gain', 'max'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('g', 'max'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('cost', 'min'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('c', 'min'), self.objectives))

        self.expert_range = expert_range

        # store values of caluclating ranked alternatives, mean, sd and topsis value
        self.mean_col = []
        self.sd_col = []
        self.topsis_val = []
        self.ranked_alternatives = []

        self.checkInput()

        #weights_ = self.weights.copy()

        self.data = self.normalizeData(self.data)

        self.weights = self.normalizeWeights(self.weights)

        self.isFitted = True

### TO DO: Remove tmp argument
    def transform(self):
        """performes any nesesary operation to prepare the ranking
        calculates and adds mean, standard deviation and topsis value columns to the dataframe and ranks the data
        Rises
        -----
        Exception
            if the data is not fitted
        """
        if(not self.isFitted):
            raise Exception("fit is required before transform")

        # MSD transformation
        self.calulateMean()
        self.calculateSD()
        self.topsis()

        # ranking
        self.ranked_alternatives = self.ranking()

    def inverse_transform(self, target):
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
    
    def improvement_basic(self, position, improvement, improvement_ratio):
      """given the parameters of an alternative shows how much improvement in standard deviation and mean is needed to get higher ranking
        Parameters
        ----------
        position : int
            current possition of alternative to improve
        improvement : int
            for how many positions to improve
        improvement_ratio : float
            how detailed should be the output
      """
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      if self.agg_fn == "I":
        while alternative_to_improve["AggFn"] < alternative_to_overcome["AggFn"]:
          alternative_to_improve["Mean"] += improvement_ratio
          alternative_to_improve["Std"] -= improvement_ratio
          alternative_to_improve["AggFn"] = 1-np.sqrt((1-alternative_to_improve["Mean"])*(1-alternative_to_improve["Mean"]) + alternative_to_improve["Std"]*alternative_to_improve["Std"])
      elif self.agg_fn == "A":
        while alternative_to_improve["AggFn"] < alternative_to_overcome["AggFn"]:
          alternative_to_improve["Mean"] += improvement_ratio
          alternative_to_improve["Std"] += improvement_ratio
          alternative_to_improve["AggFn"] = np.sqrt(alternative_to_improve["Mean"]*alternative_to_improve["Mean"] + alternative_to_improve["Std"]*alternative_to_improve["Std"])
      else:
        if alternative_to_overcome["Std"] > 0.5:
          while alternative_to_improve["AggFn"] < alternative_to_overcome["AggFn"]:
            alternative_to_improve["Mean"] += improvement_ratio
            alternative_to_improve["Std"] -= improvement_ratio
            alternative_to_improve["AggFn"] = np.sqrt(alternative_to_improve["Mean"]*alternative_to_improve["Mean"] + alternative_to_improve["Std"]*alternative_to_improve["Std"]) / (np.sqrt(alternative_to_improve["Mean"]*alternative_to_improve["Mean"] + alternative_to_improve["Std"]*alternative_to_improve["Std"]) + np.sqrt((1-alternative_to_improve["Mean"])*(1-alternative_to_improve["Mean"]) + alternative_to_improve["Std"]*alternative_to_improve["Std"]))
        else:
          while alternative_to_improve["AggFn"] < alternative_to_overcome["AggFn"]:
            alternative_to_improve["Mean"] += improvement_ratio
            alternative_to_improve["Std"] += improvement_ratio
            alternative_to_improve["AggFn"] = np.sqrt(alternative_to_improve["Mean"]*alternative_to_improve["Mean"] + alternative_to_improve["Std"]*alternative_to_improve["Std"]) / (np.sqrt(alternative_to_improve["Mean"]*alternative_to_improve["Mean"] + alternative_to_improve["Std"]*alternative_to_improve["Std"]) + np.sqrt((1-alternative_to_improve["Mean"])*(1-alternative_to_improve["Mean"]) + alternative_to_improve["Std"]*alternative_to_improve["Std"]))
      print("you should change standard deviation by:", alternative_to_improve["Std"] - self.data.loc[self.ranked_alternatives[position]]["Std"], "and mean by:", alternative_to_improve["Mean"] - self.data.loc[self.ranked_alternatives[position]]["Mean"])

    def improvement_features(self, position, improvement, improvement_ratio, features_to_change):
      """given the parameters of an alternative shows how much improvement in chosen parameters is needed to get higher ranking
        Parameters
        ----------
        position : int
            current possition of alternative to improve
        improvement : int
            for how many positions to improve
        improvement_ratio : float
            how detailed should be the output
        features_to_change : list
            which features should be improved (in given order)
      """
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      AggFn = alternative_to_improve["AggFn"]
      alternative_to_improve = alternative_to_improve.drop(labels = ["Mean", "Std", "AggFn"])
      feature_pointer = 0

      """
      while AggFn < alternative_to_overcome["AggFn"]:
        if feature_pointer == len(features_to_change):
          print("This set of features to change is not sufficiant to overcame that alternative")
          break
        while alternative_to_improve[features_to_change[feature_pointer]] + improvement_ratio <= 1:
          alternative_to_improve[features_to_change[feature_pointer]] += improvement_ratio
          improvements[feature_pointer] += improvement_ratio
          mean = alternative_to_improve.mean()
          std = alternative_to_improve.std()
          if self.agg_fn == "I":
            AggFn = 1-np.sqrt((1-mean)*(1-mean) + std*std)
          elif self.agg_fn == "A":
            AggFn = np.sqrt(mean*mean + std*std)
          else:
            AggFn = np.sqrt(mean*mean + std*std)/(np.sqrt((1-mean)*(1-mean) + std*std) + np.sqrt(mean*mean + std*std))
          if AggFn > alternative_to_overcome["AggFn"]:
            break
        feature_pointer += 1
        
      else:
        for i in range(len(features_to_change)):
          if(self.objectives[i] == "max"):
            improvements[i] = self.value_range[features_to_change[i]] * improvements[i]
          else:
            improvements[i] = -(self.value_range[features_to_change[i]] * improvements[i])
        print("to achive that you should change your features by this values:")
        print(improvements)
      """

      is_improvement_sayisfactory = False

      for i in features_to_change:
        alternative_to_improve[i] = 1
        mean = alternative_to_improve.mean()
        std = alternative_to_improve.std()
        if self.agg_fn == "I":
          AggFn = 1-np.sqrt((1-mean)*(1-mean) + std*std)
        elif self.agg_fn == "A":
          AggFn = np.sqrt(mean*mean + std*std)
        else:
          AggFn = np.sqrt(mean*mean + std*std)/(np.sqrt((1-mean)*(1-mean) + std*std) + np.sqrt(mean*mean + std*std))
        if AggFn < alternative_to_overcome["AggFn"]:
          continue

        alternative_to_improve[i] = 0.5
        mean = alternative_to_improve.mean()
        std = alternative_to_improve.std()
        if self.agg_fn == "I":
          AggFn = 1-np.sqrt((1-mean)*(1-mean) + std*std)
        elif self.agg_fn == "A":
          AggFn = np.sqrt(mean*mean + std*std)
        else:
          AggFn = np.sqrt(mean*mean + std*std)/(np.sqrt((1-mean)*(1-mean) + std*std) + np.sqrt(mean*mean + std*std))
        change_ratio = 0.25
        while True: # AggFn < alternative_to_overcome["AggFn"] or AggFn - alternative_to_overcome["AggFn"] < improvement_ratio:
          if AggFn < alternative_to_overcome["AggFn"]:
            alternative_to_improve[i] += change_ratio
          elif AggFn - alternative_to_overcome["AggFn"] > improvement_ratio:
            alternative_to_improve[i] -= change_ratio
          else:
            is_improvement_sayisfactory = True
            break
          change_ratio = change_ratio/2
          mean = alternative_to_improve.mean()
          std = alternative_to_improve.std()
          if self.agg_fn == "I":
            AggFn = 1-np.sqrt((1-mean)*(1-mean) + std*std)
          elif self.agg_fn == "A":
            AggFn = np.sqrt(mean*mean + std*std)
          else:
            AggFn = np.sqrt(mean*mean + std*std)/(np.sqrt((1-mean)*(1-mean) + std*std) + np.sqrt(mean*mean + std*std))
        
        if is_improvement_sayisfactory:
          alternative_to_improve -= self.data.loc[self.ranked_alternatives[position]].copy().drop(labels = ["Mean", "Std", "AggFn"])
          for j in range(len(alternative_to_improve)):
            if(alternative_to_improve[j] == 0):
              continue
            elif (self.objectives[j] == "max"):
              alternative_to_improve[j] = self.value_range[j] * alternative_to_improve[j]
            else:
              alternative_to_improve[j] = -self.value_range[j] * alternative_to_improve[j]

          self.printChanges(alternative_to_improve, features_to_change)
          self.printChangedRank(alternative_to_improve, self.ranked_alternatives[position])
          break

          for i in range(len(features_to_change)):
            print(improvements)
            if(improvements[i] == 0):
              print(improvements[0])
              continue
            elif(self.objectives[i] == "max"):
              improvements[i] = self.value_range[features_to_change[i]] * improvements[i]
            else:
              improvements[i] = -(self.value_range[features_to_change[i]] * improvements[i])
          print("to achive that you should change your features by this values:")
          print(improvements)

      else:
        print("This set of features to change is not sufficiant to overcame that alternative")


    # ---------------------------------------------------------
        # INTERNAL FUNCTIONS
    # ---------------------------------------------------------

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
        if self.expert_range is None:
            self.value_range = data.max()-data.min()
            data = (data-data.min())/(data.max()-data.min())
        else:
            c = 0
            self.value_range = []
            for col in data.columns:
                data[col] = (data[col] - self.expert_range[c][0]) / \
                    (self.expert_range[c][1]-self.expert_range[c][0])
                self.value_range.append(self.expert_range[c][1] - self.expert_range[c][0])
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
        weights = np.array([float(i)/max(weights) for i in weights])
        return weights

    def calulateMean(self):
        """calculates and ads mean column to dataframe"""
        self.data['Mean'] = self.data.mean(axis=1)

    def calculateSD(self):
        """calculates and ads standard dewiatiom column to dataframe"""
        self.data['Std'] = self.data.std(axis=1)

    def topsis(self):
        """calculates and ads topsis value column to dataframe"""
        if type(self.agg_fn) == str:
            if self.agg_fn == 'I':
                self.data['AggFn'] = 1 - np.sqrt((1-self.data['Mean'])*(
                    1-self.data['Mean'])+(self.data['Std']*self.data['Std']))
            elif self.agg_fn == 'A':
                self.data['AggFn'] = np.sqrt(
                    self.data['Mean']*self.data['Mean']+(self.data['Std']*self.data['Std']))
            elif self.agg_fn == 'R':
                self.data['AggFn'] = (np.sqrt(self.data['Mean']*self.data['Mean']+(self.data['Std']*self.data['Std'])))/(((1 - np.sqrt((1-self.data['Mean'])*(
                    1-self.data['Mean'])+(self.data['Std']*self.data['Std'])))-1)*(-1) + (np.sqrt(self.data['Mean']*self.data['Mean']+(self.data['Std']*self.data['Std']))))
        else:
            self.data['AggFn'] = self.agg_fn

    def ranking(self):
        """creates a ranking from the data based on topsis value column"""
        data__ = self.data.copy()
        data__ = data__.sort_values(by='AggFn', ascending=False)
        arranged = data__.index.tolist()
        return arranged

    def dictToObjectivesList(self, objectives_dict):
        objectives_list = []

        for col_name in self.data.columns:
            objectives_list.append(objectives_dict[col_name])
            
        return objectives_list

    def printChanges(self, dataframe, keys):
       dataframe = dataframe.to_frame()
       changes = dataframe.loc[keys]
       changes.columns = ['Change']
       display(changes)
        
    def printChangedRank(self, changes, alternative_id):
        updated_data = self.original_data.copy()
        row_to_update = updated_data.loc[alternative_id]

        for i, val in enumerate(changes):
           row_to_update[i] += val

        updated_data.loc[alternative_id] = row_to_update

        #We have to again calculate TOPSIS ranking, unfortunatelly some methods works only on self.data
        #Therefore we can't use methods, and we need to write code again
        temp_data = updated_data.copy()
        temp_data = self.normalizeData(temp_data)

        temp_data['Mean'] = temp_data.mean(axis=1)
        temp_data['Std'] = temp_data.std(axis=1)


        if type(self.agg_fn) == str:
            if self.agg_fn == 'I':
                temp_data['AggFn'] = 1 - np.sqrt((1-temp_data['Mean'])*(
                    1-temp_data['Mean'])+(temp_data['Std']*temp_data['Std']))
            elif self.agg_fn == 'A':
                temp_data['AggFn'] = np.sqrt(
                    temp_data['Mean']*temp_data['Mean']+(temp_data['Std']*temp_data['Std']))
            elif self.agg_fn == 'R':
                temp_data['AggFn'] = (np.sqrt(temp_data['Mean']*temp_data['Mean']+(temp_data['Std']*temp_data['Std'])))/(((1 - np.sqrt((1-temp_data['Mean'])*(
                    1-temp_data['Mean'])+(temp_data['Std']*temp_data['Std'])))-1)*(-1) + (np.sqrt(temp_data['Mean']*temp_data['Mean']+(temp_data['Std']*temp_data['Std']))))
        else:
            temp_data['AggFn'] = self.agg_fn
        
        updated_data['AggFn'] = temp_data['AggFn']
        updated_data = updated_data.sort_values(by='AggFn', ascending=False)
        updated_data = updated_data.drop(columns=['AggFn'])
        updated_data[updated_data.index.name] = updated_data.index
        display(updated_data.style.apply(self.highlightRows, axis = 1))
        
    def highlightRows(self, x):
        
        if x[len(x) - 1] == self.row_name:
            return['background-color: gray']*len(x)
        else:
            return['background-color: none']*len(x)
        
df = pd.read_csv("bus.csv", sep = ';', index_col = 0)
#objectives = ['max','max','min','max','min','min','min','max']
objectives = {
    "MaxSpeed" : 'max',
    "ComprPressure" : "max",
    "Blacking" : "min",
    "Torque" : "max",
    "SummerCons" : "min",
    "WinterCons" : "min",
    "OilCons" : "min",
    "HorsePower" : "max"}
buses = MSDTransformer()
buses.fit(df, None, objectives,  None)
buses.transform()

print(buses.ranked_alternatives)
buses.improvement_features(27,3,0.01, ['MaxSpeed', 'Blacking', 'SummerCons'])
