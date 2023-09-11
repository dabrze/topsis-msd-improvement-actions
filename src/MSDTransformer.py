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

    def __init__(self, agg_fn):

      self.agg_fn = agg_fn
      self.isFitted = False

    def fit(self, data, weights=None, objectives=None, expert_range=None):

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
            self.objectives = self.__dictToObjectivesList(objectives)
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

        self.topsis_val = []
        self.ranked_alternatives = []

        self.__checkInput()

        self.data = self.__normalizeData(self.data)

        self.weights = self.__normalizeWeights(self.weights)

        self.isFitted = True


    def changeAggregationFunction(self, agg_fn):
      self.agg_fn = agg_fn


    def transform(self):

        if(not self.isFitted):
            raise Exception("fit is required before transform")

        if(len(self.data.columns) == len(self.weights)):
            self.__wmstd()
            self.data['AggFn'] = self.agg_fn.TOPSISCalculation(np.mean(self.weights), self.data['Mean'], self.data['Std'])

            self.ranked_alternatives = self.__ranking()


    def plot(self):

      print("plot")


    def __checkInput(self):

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

        for i in range(self.m):
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

class TOPSISAggregationFunction():


    def TOPSISCalculation(self, w, wm, wsd):

      print(1)


    def improvement_mean(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, w):

      m_start = alternative_to_improve["Mean"]
      m_boundary = w
      if self.TOPSISCalculation(w, m_boundary, alternative_to_improve["Std"]) < alternative_to_overcome["AggFn"]:
        print("It is impossible to improve with only mean")
      else:
        change = (m_boundary - alternative_to_improve["Mean"])/2
        actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
        while True:
          if actual_aggfn > alternative_to_overcome["AggFn"]:
            if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
              alternative_to_improve["Mean"] -= change
              change = change/2
              actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
            else:
              break
          else:
            alternative_to_improve["Mean"] += change
            change = change/2
            actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
        print("You should change mean by ", alternative_to_improve["Mean"] - m_start)


    def improvement_features(self, w, alternative_to_improve, alternative_to_overcome, improvement_ratio, features_to_change, value_range, objectives):

      AggFn = alternative_to_improve["AggFn"]
      alternative_to_improve = alternative_to_improve.drop(labels = ["Mean", "Std", "AggFn"])
      improvement_start = alternative_to_improve.copy()
      feature_pointer = 0

      is_improvement_satisfactory = False

      s = np.sqrt(sum(w*w))/np.mean(w)
      for i in features_to_change:
        alternative_to_improve[i] = 1
        v = alternative_to_improve * w
        vw = (sum(v * w)/sum(w * w)) * w
        mean = np.sqrt(sum(vw*vw))/s
        std = np.sqrt(sum((v-vw)*(v-vw)))/s
        AggFn = self.TOPSISCalculation(np.mean(w), mean, std)

        if AggFn < alternative_to_overcome["AggFn"]:
          continue

        alternative_to_improve[i] = 0.5
        v = alternative_to_improve * w
        vw = (sum(v * w)/sum(w * w)) * w
        mean = np.sqrt(sum(vw*vw))/s
        std = np.sqrt(sum((v-vw)*(v-vw)))/s
        AggFn = self.TOPSISCalculation(np.mean(w), mean, std)
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
          AggFn = self.TOPSISCalculation(np.mean(w), mean, std)

        if is_improvement_satisfactory:
          alternative_to_improve -= improvement_start
          for j in range(len(alternative_to_improve)):
            if(alternative_to_improve[j] == 0):
              continue
            elif (objectives[j] == "max"):
              alternative_to_improve[j] = value_range[j] * alternative_to_improve[j]
            else:
              alternative_to_improve[j] = -value_range[j] * alternative_to_improve[j]

          #self.__printChanges(alternative_to_improve, features_to_change)

          print(alternative_to_improve)
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

class ATOPSIS(TOPSISAggregationFunction):

    def TOPSISCalculation(self, w, wm, wsd):

      return np.sqrt(wm*wm + wsd*wsd)/w


    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, w):

      std_start = alternative_to_improve["Std"]
      sd_boundary = w/2
      if self.TOPSISCalculation(w, alternative_to_improve["Mean"], sd_boundary) < alternative_to_overcome["AggFn"]:
         print("It is impossible to improve with only standard deviation")
      else:
         change = (sd_boundary - alternative_to_improve["Std"])/2
         actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
         while True:
           if actual_aggfn > alternative_to_overcome["AggFn"]:
             if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
               alternative_to_improve["Std"] -= change
               change = change/2
               actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
             else:
               break
           else:
             alternative_to_improve["Std"] += change
             change = change/2
             actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
         print("You should change standard deviation by ", alternative_to_improve["Std"] - std_start)

class ITOPSIS(TOPSISAggregationFunction):

    def TOPSISCalculation(self, w, wm, wsd):

      return np.sqrt(wm*wm + wsd*wsd)/(np.sqrt(wm*wm + wsd*wsd) + np.sqrt((w-wm) * (w-wm) + wsd*wsd))


    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, w):

      std_start = alternative_to_improve["Std"]
      sd_boundary = w/2
      if self.TOPSISCalculation(w, alternative_to_improve["Mean"], 0) < alternative_to_overcome["AggFn"]:
        print("It is impossible to improve with only standard deviation")
      else:
        change = alternative_to_improve["Std"]/2
        actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
        while True:
         if actual_aggfn > alternative_to_overcome["AggFn"]:
           if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
              alternative_to_improve["Std"] += change
              change = change/2
              actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
           else:
              break
         else:
             alternative_to_improve["Std"] -= change
             change = change/2
             actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
         print("You should change standard deviation by ", std_start - alternative_to_improve["Std"])

class RTOPSIS(TOPSISAggregationFunction):

    def TOPSISCalculation(self, w, wm, wsd):

      return np.sqrt(wm*wm + wsd*wsd)/(np.sqrt(wm*wm + wsd*wsd) + np.sqrt((w-wm) * (w-wm) + wsd*wsd))


    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, w):

      std_start = alternative_to_improve["Std"]
      sd_boundary = w/2
      if (alternative_to_improve["Mean"]<sd_boundary):
        if self.TOPSISCalculation(w, alternative_to_improve["Mean"], sd_boundary) < alternative_to_overcome["AggFn"]:
          print("It is impossible to improve with only standard deviation")
        else:
          change = (sd_boundary - alternative_to_improve["Std"])/2
          actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
          while True:
            if actual_aggfn > alternative_to_overcome["AggFn"]:
              if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                alternative_to_improve["Std"] -= change
                change = change/2
                actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
              else:
                break
            else:
              alternative_to_improve["Std"] += change
              change = change/2
              actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
          print("You should change standard deviation by ", alternative_to_improve["Std"] - std_start)
      else:
        if self.TOPSISCalculation(w, alternative_to_improve["Mean"], 0) < alternative_to_overcome["AggFn"]:
          print("It is impossible to improve with only standard deviation")
        else:
          change = alternative_to_improve["Std"]/2
          actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
          while True:
            if actual_aggfn > alternative_to_overcome["AggFn"]:
              if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                alternative_to_improve["Std"] += change
                change = change/2
                actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
              else:
                break
            else:
              alternative_to_improve["Std"] -= change
              change = change/2
              actual_aggfn = self.TOPSISCalculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
          print("You should change standard deviation by ", std_start - alternative_to_improve["Std"])

