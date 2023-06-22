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
    A class used to: calculate TOPSIS ranking,
    plot positions of alternatives in MSD space,
    perform improvement actions on selected alternative.
    
    ...
    Attributes
    ----------
    original_data : dataframe
        Pandas dataframe provided by the user.
    data : dataframe
        A copy of self.original_data, on which all calculations are performed.
    n : int
        Number of dataframe's columns
    m : int
        Number of dataframe's rows
    original_weights : np.array of float
        Numpy array of criteria' weights.
    weights : np.array of float, optional
        Normalized self.original_weights.
    objectives : np.array of str
        Numpy array informing which criteria are cost type
        and which are gain type.
    expert_range : none
        TO DO
    isFitted : bool
        Simple flag which takes True value only when the fit() method
        was performed on MSDTransformer object.
    topsis_val : list of float
        List of calculated TOPSIS values of self.dataframe.
    ranked_alternatives : list of str
        List of alternatives' ID's ordered according to their TOPSIS values.
    """

    # ---------------------------------------------------------
    # EXTERNAL FUNCTIONS
    # ---------------------------------------------------------

    def __init__(self, agg_fn='I'):
        """
        Parameters
        ----------
        agg_fn : string or function, optional
            Aggregation function which is used to calculate the TOPSIS value. 
            It can be passed as the custom function, or take one of the following values:
            "I", "A", "R".
        """

        self.agg_fn = agg_fn
        self.isFitted = False

    def fit(self, data, weights=None, objectives=None, expert_range=None):
        """Checks input data and normalizes it.
        Parameters
        ----------
        data : dataframe
            Pandas dataframe provided by the user. 
            Apart of column and row names all values must be numerical.
        weights : np.array of float, optional
            Numpy array of criteria' weights. 
            Its length must be equal to self.n.
            (default: np.ones())
        objectives : list or dict or str, optional
            Numpy array informing which criteria are cost type and which are gain type.
            It can be passed as:
            - list of length equal to self.n. in which each element describes type of one criterion:
            'cost'/'c'/'min' for cost type criteria and 'gain'/'g'/'max' for gain type criteria.
            - dictionary of size equal to self.n in which each key is the criterion name and ech value takes one of the following values:
            'cost'/'c'/'min' for cost type criteria and 'gain'/'g'/'max' for gain type criteria.
            - a string which describes type of all criteria:
            'cost'/'c'/'min' if criteria are cost type and 'gain'/'g'/'max' if criteria are gain type.
            (default: list of 'max')
        expert_range : none, optional
            TO DO
        """

        self.original_data = data

        self.data = self.original_data.copy()
        self.n = self.data.shape[1]
        self.m = self.data.shape[0]

        self.original_weights = (weights if weights is not None else np.ones(self.n))
        self.weights = self.original_weights.copy()

        self.objectives = objectives

        if(type(objectives) is list):
            self.objectives = objectives
        elif(type(objectives) is str):
            self.objectives = np.repeat(objectives, self.n)
        elif(type(objectives) is dict):
            self.objectives = self.__dictToObjectivesList(objectives)
        elif(objectives is None):
            self.objectives = np.repeat('max', self.n)

        self.objectives = list(
            map(lambda x: x.replace('gain', 'max'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('g', 'max'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('cost', 'min'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('c', 'min'), self.objectives))

        self.expert_range = expert_range

        #self.mean_col = []
        #self.sd_col = []
        self.topsis_val = []
        self.ranked_alternatives = []

        self.__checkInput()

        self.data = self.__normalizeData(self.data)

        self.weights = self.__normalizeWeights(self.weights)

        self.isFitted = True

    def transform(self):
        """ Adds to self.data 'Mean', 'Std' and 'AggFn' columns,
        which contain values of mean, standard deviation and TOPSIS for every alternative.
        Based on calculated TOPSIS values it ranks alternatives and saves their order in self.ranked_alternatives.
        
        Rises
        -----
        Exception
            If on the MSDTransport object wasn't performed fit() method.
        """
        if(not self.isFitted):
            raise Exception("fit is required before transform")

        if(len(self.data.columns) == len(self.weights)):
            self.__wmstd()
            #self.__calculateMean()
            #self.__calculateSD()
            self.__topsis()

            self.ranked_alternatives = self.__ranking()

    def inverse_transform(self, target):
        """ TO DO
        Parameters
        ----------
        target : none
            TO DO
        Returns
        -------
        TO DO
        """
        target_ = target.copy()

        return target_

    def plot(self):
        """ Plots positions of alternatives in MSD space.
        """
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
        threshold = 0.015
        indexes = temp_DataFrame.index.tolist()
        isGood = False
        
        for k in range(1,len(temp_DataFrame)-1):
            j=k
            if(not isGood):

                isGood = True

                for i in range(k,len(temp_DataFrame)-1):
        
                    Dif_B = temp_DataFrame['Std'][indexes[j]] - temp_DataFrame['Std'][indexes[j-1]]
                    Dif_A = temp_DataFrame['Std'][indexes[j]] - temp_DataFrame['Std'][indexes[j+1]]
                    

                    if ((Dif_A>=0 or Dif_B>=0) or ((Dif_B >= (-1)*threshold and Dif_A >= (-1)*threshold))):
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
    
    def improvement_mean(self, position, improvement, improvement_ratio):
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      m_boundary = np.mean(self.weights)
      if self.agg(m_boundary, alternative_to_improve["Std"]) < alternative_to_overcome["AggFn"]:
        print("It is impossible to improve with only mean")
      else:
        change = (m_boundary - alternative_to_improve["Mean"])/2
        actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
        while True:
          if actual_aggfn > alternative_to_overcome["AggFn"]:
            if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
              alternative_to_improve["Mean"] -= change
              change = change/2
              actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
            else:
              break
          else:
            alternative_to_improve["Mean"] += change
            change = change/2
            actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
        print("You should change mean by ", alternative_to_improve["Mean"] - self.data.loc[self.ranked_alternatives[position]]["Mean"])

    def improvement_std(self, position, improvement, improvement_ratio):
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      sd_boundary = np.mean(self.weights)/2
      if (self.agg_fn == "A") or (self.agg_fn == "R" and alternative_to_improve["Mean"]<sd_boundary):
        if self.agg(alternative_to_improve["Mean"], sd_boundary) < alternative_to_overcome["AggFn"]:
          print("It is impossible to improve with only standard deviation")
        else:
          change = (sd_boundary - alternative_to_improve["Std"])/2
          actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          while True:
            if actual_aggfn > alternative_to_overcome["AggFn"]:
              if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                alternative_to_improve["Std"] -= change
                change = change/2
                actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
              else:
                break
            else:
              alternative_to_improve["Std"] += change
              change = change/2
              actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          print("You should change standard deviation by ", alternative_to_improve["Std"] - self.data.loc[self.ranked_alternatives[position]]["Std"])
      else:
        if self.agg(alternative_to_improve["Mean"], 0) < alternative_to_overcome["AggFn"]:
          print("It is impossible to improve with only standard deviation")
        else:
          change = alternative_to_improve["Std"]/2
          actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          while True:
            if actual_aggfn > alternative_to_overcome["AggFn"]:
              if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                alternative_to_improve["Std"] += change
                change = change/2
                actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
              else:
                break
            else:
              alternative_to_improve["Std"] -= change
              change = change/2
              actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          print("You should change standard deviation by ", self.data.loc[self.ranked_alternatives[position]]["Std"] - alternative_to_improve["Std"])

    def improvement_full(self, position, improvement, improvement_ratio):
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      m_boundary = np.mean(self.weights)
      change_m = (m_boundary - alternative_to_improve["Mean"])/2
      change_sd = alternative_to_improve["Std"]/2
      actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
      while True:
        if actual_aggfn > alternative_to_overcome["AggFn"]:
          if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
            alternative_to_improve["Std"] += change_sd
            alternative_to_improve["Mean"] -= change_m
            change_sd = change_sd/2
            change_m = change_m/2
            actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          else:
            break
        else:
          alternative_to_improve["Std"] -= change_sd
          alternative_to_improve["Mean"]+= change_m
          change_sd = change_sd/2
          change_m = change_m/2
          actual_aggfn = self.agg(alternative_to_improve["Mean"], alternative_to_improve["Std"])
      print("You should change standard deviation by ", self.data.loc[self.ranked_alternatives[position]]["Std"] - alternative_to_improve["Std"]," and mean by ", alternative_to_improve["Mean"] - self.data.loc[self.ranked_alternatives[position]]["Mean"])

    
    def improvement_basic(self, position, improvement, improvement_ratio):
      """Calculates minimal change of mean and standard deviation,
      needed to change a rank of given alternative.
        
        Parameters
        ----------
        position : int
            TO DO
        improvement : int
            TO DO
        improvement_ratio : float
            TO DO
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
      """Calculates minimal change of the criteria,
      needed to change a rank of given alternative.
        Parameters
        ----------
        position : int
            TO DO
        improvement : int
            TO DO
        improvement_ratio : float
            TO DO
        features_to_change : list
            TO DO
      """
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      AggFn = alternative_to_improve["AggFn"]
      alternative_to_improve = alternative_to_improve.drop(labels = ["Mean", "Std", "AggFn"])
      feature_pointer = 0

      is_improvement_satisfactory = False

      w = self.weights
      s = np.sqrt(sum(w*w))/np.mean(w)
      for i in features_to_change:
        alternative_to_improve[i] = 1
        v = alternative_to_improve * w
        vw = (sum(v * w)/sum(w * w)) * w
        mean = np.sqrt(sum(vw*vw))/s
        std = np.sqrt(sum((v-vw)*(v-vw)))/s
        AggFn = self.agg(mean, std)
        
        if AggFn < alternative_to_overcome["AggFn"]:
          continue

        alternative_to_improve[i] = 0.5
        v = alternative_to_improve * w
        vw = (sum(v * w)/sum(w * w)) * w
        mean = np.sqrt(sum(vw*vw))/s
        std = np.sqrt(sum((v-vw)*(v-vw)))/s
        AggFn = self.agg(mean, std)
        change_ratio = 0.25
        while True: 
          if AggFn < alternative_to_overcome["AggFn"]:
            alternative_to_improve[i] += change_ratio
          elif AggFn - alternative_to_overcome["AggFn"] > improvement_ratio:
            alternative_to_improve[i] -= change_ratio
          else:
            is_improvement_satisfactory = True
            break
          change_ratio = change_ratio/2
          v = alternative_to_improve * w
          vw = (sum(v * w)/sum(w * w)) * w
          mean = np.sqrt(sum(vw*vw))/s
          std = np.sqrt(sum((v-vw)*(v-vw)))/s
          AggFn = self.agg(mean, std)
        
        if is_improvement_satisfactory:
          alternative_to_improve -= self.data.loc[self.ranked_alternatives[position]].copy().drop(labels = ["Mean", "Std", "AggFn"])
          for j in range(len(alternative_to_improve)):
            if(alternative_to_improve[j] == 0):
              continue
            elif (self.objectives[j] == "max"):
              alternative_to_improve[j] = self.value_range[j] * alternative_to_improve[j]
            else:
              alternative_to_improve[j] = -self.value_range[j] * alternative_to_improve[j]

          self.__printChanges(alternative_to_improve, features_to_change)
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
          print("to achieve that you should change your features by this values:")
          print(improvements)

      else:
        print("This set of features to change is not sufficient to overcame that alternative")


    # ---------------------------------------------------------
        # INTERNAL FUNCTIONS
    # ---------------------------------------------------------

    def __checkInput(self):
      
        if (len(self.weights) != self.n):
            raise ValueError("Invalid value 'weights'.")

        if(not all(type(item) in [int, float, np.float64] for item in self.weights)):
            raise ValueError("Invalid value 'weights'. Expected numerical value (int or float).")

        if (len(self.objectives) != self.n):
            raise ValueError("Invalid value 'objectives'.")

        if(not all(item in ["min", "max"] for item in self.objectives)):
            raise ValueError(
                "Invalid value at 'objectives'. Use 'min', 'max', 'gain', 'cost', 'g' or 'c'.")

        if(self.expert_range != None):
            if(len(self.expert_range) != len(self.objectives)):
                raise ValueError(
                    "Invalid value at 'expert_range'. Length of should be equal to number of criteria.")
            for col in self.expert_range:
                if(len(col) != 2):
                    raise ValueError(
                        "Invalid value at 'expert_range'. Every criterion has to have minimal and maximal value.")
                if(not all(type(item) in [int, float] for item in col)):
                    raise ValueError(
                        "Invalid value at 'expert_range'. Expected numerical value (int or float).")
                if(col[0] > col[1]):
                    raise ValueError("Invalid value at 'expert_range'. Minimal value  is bigger then maximal value.")

    def __normalizeData(self, data):
        """normalize given data using either given expert range or min/max
        uses the min-max normalization with minimum and maximum taken from expert ranges if given
        Parameters
        ----------
        data : dataframe
            data to be normalized
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

        for i in range(self.n):
            if self.objectives[i] == 'min':
                data[data.columns[i]] = 1 - data[data.columns[i]]

        return data

    def __normalizeWeights(self, weights):
        """normalize weights
        result are weights not greater than 1 but not 0 if not present previously
        Parameters
        ----------
        weights : np.array
            weights to be normalized
        """
        weights = np.array([float(i)/max(weights) for i in weights])
        return weights

    def __wmstd(self):

      w = self.weights
      s = np.sqrt(sum(w*w))/np.mean(w)
      wm = []
      wsd = []
      for index, row in self.data.iterrows():
        v = row * w
        vw = (sum(v * w)/sum(w * w)) * w
        wm.append(np.sqrt(sum(vw*vw))/s)
        wsd.append(np.sqrt(sum((v-vw)*(v-vw)))/s)

      self.data['Mean'] = wm
      self.data['Std'] = wsd



    def __calculateMean(self):
        """calculates and adds mean column to dataframe"""
        self.data['Mean'] = self.data.mean(axis=1)

    def __calculateSD(self):
        """calculates and adds standard deviation column to dataframe"""
        self.data['Std'] = self.data.std(axis=1)

    def agg(self, wm, wsd):
        w = np.mean(self.weights)
        if self.agg_fn == 'I':
            return 1 - np.sqrt((w-wm) * (w-wm) + wsd*wsd)/w
        elif self.agg_fn == 'A':
            return np.sqrt(wm*wm + wsd*wsd)/w
        elif self.agg_fn == 'R':
            return np.sqrt(wm*wm + wsd*wsd)/(np.sqrt(wm*wm + wsd*wsd) + np.sqrt((w-wm) * (w-wm) + wsd*wsd))

    def __topsis(self):
        """calculates and adds topsis value column to dataframe"""

        self.data['AggFn'] = self.agg(self.data['Mean'], self.data['Std'])

        


        '''if type(self.agg_fn) == str:
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
            self.data['AggFn'] = self.agg_fn'''



    def __ranking(self):
        """creates a ranking from the data based on topsis value column"""
        data__ = self.data.copy()
        data__ = data__.sort_values(by='AggFn', ascending=False)
        arranged = data__.index.tolist()
        return arranged

    def __dictToObjectivesList(self, objectives_dict):
        objectives_list = []

        for col_name in self.data.columns:
            objectives_list.append(objectives_dict[col_name])
            
        return objectives_list

    def __printChanges(self, dataframe, keys):
        dataframe = dataframe.to_frame()
        changes = dataframe.loc[keys]
        changes.columns = ['Change']
        self.improvement = changes
        display(changes)
    
    def __printChangedRank(self, changes, alternative_id):
        updated_data = self.original_data.copy()
        row_to_update = updated_data.loc[alternative_id]

        for i, val in enumerate(changes):
           row_to_update[i] += val

        updated_data.loc[alternative_id] = row_to_update

        temp_data = updated_data.copy()
        temp_data = self.__normalizeData(temp_data)

        w = self.weights
        s = np.sqrt(sum(w*w))/np.mean(w)
        wm = []
        wsd = []
        for index, row in self.data.iterrows():
          v = row * w
          vw = (sum(v * w)/sum(w * w)) * w
          wm.append(np.sqrt(sum(vw*vw))/s)
          wsd.append(np.sqrt(sum((v-vw)*(v-vw)))/s)

        temp_data['Mean'] = wm
        temp_data['Std'] = wsd


        temp_data['AggFn'] = self.agg(self.data['Mean'], self.data['Std'])
        
        updated_data['AggFn'] = temp_data['AggFn']
        updated_data = updated_data.sort_values(by='AggFn', ascending=False)
        updated_data = updated_data.drop(columns=['AggFn'])
        updated_data[updated_data.index.name] = updated_data.index
        display(updated_data.style.apply(self.__highlightRows, axis = 1))
        
    def __highlightRows(self, x):
        if x[len(x) - 1] == self.row_name:
            return['background-color: gray']*len(x)
        else:
            return['background-color: none']*len(x)
