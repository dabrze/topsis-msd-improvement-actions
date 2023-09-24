from abc import ABC, abstractmethod
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
        self.agg_fn = self.__check_agg_fn(agg_fn)
        self.isFitted = False
             
    def fit(self, X, weights=None, objectives=None, expert_range=None):

        self.X = X
        self.m = X.shape[1]
        self.n = X.shape[0]

        self.original_weights = self.__check_weights(weights)
        self.weights = self.original_weights.copy()

        self.objectives = self.__check_objectives(objectives)

        self.objectives = list(
            map(lambda x: x.replace('gain', 'max'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('g', 'max'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('cost', 'min'), self.objectives))
        self.objectives = list(
            map(lambda x: x.replace('c', 'min'), self.objectives))

        self.expert_range = self.__check_expert_range(expert_range)

        self.topsis_val = []
        self.ranked_alternatives = []

        self.__checkInput()

        self.weights = self.__normalizeWeights(self.weights)

        self.isFitted = True

        return self

    def changeAggregationFunction(self, agg_fn):
        self.agg_fn = self.__check_agg_fn(agg_fn)

    def transform(self, X):

        if(not self.isFitted):
            raise Exception("fit is required before transform")

        X_new = X.copy()
        self.X_new = self.__normalizeData(X_new)

        if(len(self.X_new.columns) == len(self.weights)):
            self.__wmstd()
            self.X_new['AggFn'] = self.agg_fn.TOPSISCalculation(np.mean(self.weights), self.X_new['Mean'], self.X_new['Std'])

            self.ranked_alternatives = self.__ranking()
            
        return self.X_new

    def transform_new_data(self, X, normalize_data=False, print_ranks=False):
        if not self.isFitted:
            raise Exception("fit is required before transforming new data")

        if normalize_data:
            # TODO Make sure that the performances of the new alternatives does not exceed the initially established range of evaluations
            X_US = (np.array(X) - np.array(self.lower_bounds)) / np.array(self.value_range)

            for i in range(self.m):
                if self.objectives[i] == 'min':
                    X_US[:, i] = 1 - X_US[:, i]
        else:
            X_US = np.array(X)

        w_means, w_stds = self.transform_US_to_wmsd(X_US)
        agg_values = self.agg_fn.TOPSISCalculation(np.mean(self.weights), w_means, w_stds)
        if print_ranks:
            ranking_func = np.vectorize(lambda agg_value: 1 + np.sum(self.X_new['AggFn'] > agg_value))
            ranks = ranking_func(agg_values)
            print(agg_values, ranks)
        return w_means, w_stds, agg_values

    def transform_US_to_wmsd(self, X_US):
        # transform data from Utility Space to WMSD Space
        w = self.weights
        s = np.linalg.norm(w) / np.mean(w)
        v = X_US * w

        vw = (np.sum(v * w, axis=1) / np.sum(w ** 2)).reshape(-1, 1) @ w.reshape(1, -1)
        wmeans = np.linalg.norm(vw, axis=1) / s
        wstds = np.linalg.norm(v - vw, axis=1) / s
        return wmeans, wstds

    def improvement(self, function_name, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):

        if(type(alternative_to_improve) == int):
          alternative_to_improve = self.X_new.loc[self.ranked_alternatives[alternative_to_improve]].copy()
        elif(type(alternative_to_improve) == str):
          alternative_to_improve = self.X_new.loc[alternative_to_improve].copy()

        if(type(alternative_to_overcome) == int):
          alternative_to_overcome = self.X_new.loc[self.ranked_alternatives[alternative_to_overcome]].copy()
        elif(type(alternative_to_overcome) == str):
          alternative_to_overcome = self.X_new.loc[alternative_to_overcome].copy()
          

        func = getattr(self.agg_fn, function_name)
        func(alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs)

    def plot(self):

      print("plot")

    def show_ranking(self, mode = None, first = 1, last = None):

        if last is None:
           last = len(self.X_new.index)

        self.__check_show_ranking(first, last)

        ranking = self.X_new
        ranking = ranking.assign(Rank = None)
        columns = ranking.columns.tolist()
        columns = columns[-1:] + columns[:-1]
        ranking = ranking[columns]

        alternative_names = ranking.index.tolist()
        for alternative in alternative_names:
            ranking['Rank'][alternative] = self.ranked_alternatives.index(alternative) + 1

        ranking = ranking.sort_values(by = ['Rank'])
        #ranking = ranking.loc[max(first-1, 0):last]
        ranking = ranking[(first-1):last]

        if isinstance(mode, str):
            if mode == 'minimal':
                display(ranking['Rank'])
            elif mode == 'standard':
                display(ranking.drop(['Mean', 'Std', 'AggFn'], axis=1))
            elif mode == 'full':
                display(ranking)
            else:
               raise ValueError("Invalid value at 'mode': must be a string (minimal, standard, or full).")
            return
        
        display(ranking.drop(['Mean', 'Std', 'AggFn'], axis=1))
        return
    
    def __check_agg_fn(self, agg_fn):
        if isinstance(agg_fn, str):
            if agg_fn == "A":
                return ATOPSIS(self)
            elif agg_fn == "I":
                return ITOPSIS(self)
            elif agg_fn == "R":
                return RTOPSIS(self)
            else:
                raise ValueError("Invalid value at 'agg_fn': must be string (A, I, or R) or class implementing TOPSISAggregationFunction.")
        elif issubclass(agg_fn, TOPSISAggregationFunction):
            return agg_fn(self)
        else:
            raise ValueError("Invalid value at 'agg_fn': must be string (A, I, or R) or class implementing TOPSISAggregationFunction.")
  
    def __check_weights(self, weights):
        if isinstance(weights, list):
           return weights
        
        elif isinstance(weights, dict):
           return self.__dictToList(weights)
        
        elif weights is None:
           return np.ones(self.m)
        
        else:
           raise ValueError("Invalid value at 'weights': must be a list or a dictionary")

    def __check_objectives(self, objectives):
        if isinstance(objectives, list):
           return objectives
        elif isinstance(objectives, str):
           return np.repeat(objectives, self.m)
        elif isinstance(objectives, dict):
           return self.__dictToList(objectives)
        elif objectives is None:
           return np.repeat('max', self.m)
        else:
           raise ValueError("Invalid value at 'objectives': must be a list or a string (gain, g, cost, c, min or max) or a dictionary")

    def __check_expert_range(self, expert_range):
        if isinstance(expert_range, dict):
            expert_range = self.__dictToList(expert_range)

        if isinstance(expert_range, list):

            if all(isinstance(e, list) for e in expert_range):
                return expert_range
            
            elif all(isinstance(e, (int, float, np.float64)) for e in expert_range):
                expert_range = [expert_range]
                numpy_expert_range = np.repeat(expert_range, self.m, axis = 0)
                return numpy_expert_range.tolist()
            
            else:
               raise ValueError("Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary")

        elif expert_range is None:
           print(3)
           return
        
        else:
           raise ValueError("Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary")
        
    def __checkInput(self):

        if (len(self.weights) != self.m):
            raise ValueError("Invalid value 'weights'.")

        if(not all(type(item) in [int, float, np.float64] for item in self.weights)):
            raise ValueError("Invalid value 'weights'. Expected numerical value (int or float).")
        
        if(not all(item >= 0 for item in self.weights)):
            raise ValueError("Invalid value 'weights'. Expected value must be non-negative.")
        
        if(not any(item > 0 for item in self.weights)):
            raise ValueError("Invalid value 'weights'. At least one weight must be positive.")

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

    def __check_show_ranking(self, first, last):

        if isinstance(first, int):
           if first < 1 or first > len(self.X_new.index):
              raise ValueError(f"Invalid value at 'first': must be in range [1:{len(self.X_new.index)}]")
        else:
           raise TypeError("Invalid type of 'first': must be an int")
        
        if isinstance(last, int):
           if last < 1 or last > len(self.X_new.index):
              raise ValueError(f"Invalid value at 'last': must be in range [1:{len(self.X_new.index)}]")
        else:
           raise TypeError("Invalid type of 'last': must be an int")
        
        if last < first:
           raise ValueError("'first' must be not greater than 'last'")
        
    def __normalizeData(self, data):
        """normalize given data using either given expert range or min/max
        uses the min-max normalization with minimum and maximum taken from expert ranges if given
        Parameters
        ----------
        data : dataframe
            data to be normalized
        """
        if self.expert_range is None:
            self.lower_bounds = data.min()
            self.value_range = data.max()-data.min()
            data = (data-data.min())/(data.max()-data.min())
        else:
            c = 0
            self.value_range = []
            self.lower_bounds = []
            for col in data.columns:
                data[col] = (data[col] - self.expert_range[c][0]) / \
                    (self.expert_range[c][1]-self.expert_range[c][0])
                self.value_range.append(self.expert_range[c][1] - self.expert_range[c][0])
                self.lower_bounds.append(self.expert_range[c][0])
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
      for index, row in self.X_new.iterrows():
        v = row * w
        vw = (sum(v * w)/sum(w * w)) * w
        wm.append(np.sqrt(sum(vw*vw))/s)
        wsd.append(np.sqrt(sum((v-vw)*(v-vw)))/s)

      self.X_new['Mean'] = wm
      self.X_new['Std'] = wsd

    def __ranking(self):
        """creates a ranking from the data based on topsis value column"""
        data__ = self.X_new.copy()
        data__ = data__.sort_values(by='AggFn', ascending=False)
        arranged = data__.index.tolist()
        return arranged

    def __dictToList(self, dictionary):
        new_list = []

        for col_name in self.X.columns:
            new_list.append(dictionary[col_name])

        return new_list

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
        for index, row in self.X_new.iterrows():
          v = row * w
          vw = (sum(v * w)/sum(w * w)) * w
          wm.append(np.sqrt(sum(vw*vw))/s)
          wsd.append(np.sqrt(sum((v-vw)*(v-vw)))/s)

        temp_data['Mean'] = wm
        temp_data['Std'] = wsd


        temp_data['AggFn'] = self.agg(self.X_new['Mean'], self.X_new['Std'])

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


class TOPSISAggregationFunction(ABC):
    def __init__(self, msd_transformer):
        self.msd_transformer = msd_transformer

    @abstractmethod
    def TOPSISCalculation(self, w, wm, wsd):
        pass

    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change, alternative_to_improve_CS, **kwargs):
        """ Universal binary search algorithm for achieving the target by modifying the performance on a single criterion. """
        performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        target_agg_value = alternative_to_overcome["AggFn"] + improvement_ratio

        modified_criterion_idx = list(alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).index).index(feature_to_change)
        criterion_range = self.msd_transformer.value_range[modified_criterion_idx]

        max_possible_improved = performances_US.copy()
        max_possible_improved[modified_criterion_idx] = 1
        max_possible_agg_value = self.msd_transformer.transform_new_data([max_possible_improved])[2].item()
        if max_possible_agg_value < target_agg_value:
            # print(f"Not possible to achieve target {target_agg_value}. Max possible agg value is {max_possible_agg_value}")
            return None

        low = 0
        high = 1
        while high - low > 1e-15:
            mid = (high + low) / 2
            improved = performances_US.copy()
            improved[modified_criterion_idx] = mid
            agg_value = self.msd_transformer.transform_new_data([improved])[2].item()

            if agg_value < target_agg_value:
                low = mid
            elif agg_value > target_agg_value:
                high = mid
            else:
                # print("while loop breaks: the exact value we were looking for has been found")
                break
        else:
            # print("while loop terminates: high ≈ low")
            pass

        improvement_CS = (mid - performances_US[modified_criterion_idx]) * criterion_range
        # print(feature_to_change, "needs to be improved by", improvement_CS)
        return improvement_CS

    def improvement_mean(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):

      w = np.mean(self.msd_transformer.weights)
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


    def improvement_features(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, features_to_change, **kwargs):
      
      AggFn = alternative_to_improve["AggFn"]
      alternative_to_improve = alternative_to_improve.drop(labels = ["Mean", "Std", "AggFn"])
      improvement_start = alternative_to_improve.copy()
      feature_pointer = 0
      w = self.msd_transformer.weights
      value_range = self.msd_transformer.value_range
      objectives = self.msd_transformer.objectives

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
          display(alternative_to_improve.to_frame(name = "Improvement rate"))
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

    @staticmethod
    def solve_quadratic_equation(a, b, c):
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        solution_1 = (-b + np.sqrt(discriminant)) / (2 * a)
        solution_2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return solution_1, solution_2

    @staticmethod
    def choose_appropriate_solution(solution_1, solution_2, lower_bound, upper_bound, objective):
        solution_1_is_feasible = upper_bound > solution_1 > lower_bound
        solution_2_is_feasible = upper_bound > solution_2 > lower_bound
        if solution_1_is_feasible:
            if solution_2_is_feasible:
                # print("Both solutions feasible")
                if objective == "max":
                    return min(solution_1, solution_2)
                else:
                    return max(solution_1, solution_2)
            else:
                # print("Only solution_1 is feasible")
                return solution_1
        else:
            if solution_2_is_feasible:
                # print("Only solution_2 is feasible")
                return solution_2
            else:
                # print("Neither solution is feasible")
                return None

class ATOPSIS(TOPSISAggregationFunction):
    def __init__(self, msd_transformer):
        super().__init__(msd_transformer)

    def TOPSISCalculation(self, w, wm, wsd):

      return np.sqrt(wm*wm + wsd*wsd)/w

    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change,
                                   alternative_to_improve_CS, **kwargs):
        """ Exact algorithm dedicated to the aggregation `A` for achieving the target by modifying the performance on a single criterion. """
        performances_CS = alternative_to_improve_CS.to_numpy().copy()
        performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        weights = self.msd_transformer.weights
        target_agg_value = (alternative_to_overcome["AggFn"] + improvement_ratio) * np.linalg.norm(weights)

        modified_criterion_idx = list(alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).index).index(feature_to_change)
        criterion_range = self.msd_transformer.value_range[modified_criterion_idx]
        lower_bound = self.msd_transformer.lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.msd_transformer.objectives[modified_criterion_idx]

        # Negative Ideal Solution (utility space)
        NIS = np.zeros_like(performances_US)

        v_ij = performances_US * weights
        j = modified_criterion_idx

        v_ij_excluding_j = np.delete(v_ij, j)
        NIS_excluding_j = np.delete(NIS, j)

        a = 1
        b = -2 * NIS[j]
        c = NIS[j] ** 2 + np.sum((v_ij_excluding_j - NIS_excluding_j) ** 2) - target_agg_value ** 2

        solutions = TOPSISAggregationFunction.solve_quadratic_equation(a, b, c)  # solutions are new performances in VS, not modifications
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = ((solutions[0] / weights[j]) * criterion_range) + lower_bound
            solution_2 = ((solutions[1] / weights[j]) * criterion_range) + lower_bound

            # solution -- new performances in CS
            solution = TOPSISAggregationFunction.choose_appropriate_solution(solution_1, solution_2, lower_bound, upper_bound, objective)
            if solution is None:
                return None
            else:
                return solution - performances_CS[j]

    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):

      w = np.mean(self.msd_transformer.weights)
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
    def __init__(self, msd_transformer):
        super().__init__(msd_transformer)

    def TOPSISCalculation(self, w, wm, wsd):
        return 1 - np.sqrt((w - wm) * (w - wm) + wsd * wsd) / w

    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change,
                                   alternative_to_improve_CS, **kwargs):
        """ Exact algorithm dedicated to the aggregation `I` for achieving the target by modifying the performance on a single criterion. """
        performances_CS = alternative_to_improve_CS.to_numpy().copy()
        performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        weights = self.msd_transformer.weights
        target_agg_value = (1 - (alternative_to_overcome["AggFn"] + improvement_ratio)) * np.linalg.norm(weights)

        modified_criterion_idx = list(alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).index).index(feature_to_change)
        criterion_range = self.msd_transformer.value_range[modified_criterion_idx]
        lower_bound = self.msd_transformer.lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.msd_transformer.objectives[modified_criterion_idx]

        # Positive Ideal Solution (utility space)
        PIS = weights

        v_ij = performances_US * weights
        j = modified_criterion_idx

        v_ij_excluding_j = np.delete(v_ij, j)
        PIS_excluding_j = np.delete(PIS, j)

        a = 1
        b = -2 * PIS[j]
        c = PIS[j] ** 2 + np.sum((v_ij_excluding_j - PIS_excluding_j) ** 2) - target_agg_value ** 2

        solutions = TOPSISAggregationFunction.solve_quadratic_equation(a, b, c)  # solutions are new performances in VS, not modifications
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = ((solutions[0] / weights[j]) * criterion_range) + lower_bound
            solution_2 = ((solutions[1] / weights[j]) * criterion_range) + lower_bound

            # solution -- new performances in CS
            solution = TOPSISAggregationFunction.choose_appropriate_solution(solution_1, solution_2, lower_bound, upper_bound, objective)
            if solution is None:
                return None
            else:
                return solution - performances_CS[j]

    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):

      w = np.mean(self.msd_transformer.weights)
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
    def __init__(self, msd_transformer):
        super().__init__(msd_transformer)

    def TOPSISCalculation(self, w, wm, wsd):

      return np.sqrt(wm*wm + wsd*wsd)/(np.sqrt(wm*wm + wsd*wsd) + np.sqrt((w-wm) * (w-wm) + wsd*wsd))

    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change,
                                   alternative_to_improve_CS, , **kwargs):
        """ Exact algorithm dedicated to the aggregation `R` for achieving the target by modifying the performance on a single criterion. """
        performances_CS = alternative_to_improve_CS.to_numpy().copy()
        performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        weights = self.msd_transformer.weights
        target_agg_value = alternative_to_overcome["AggFn"] + improvement_ratio

        modified_criterion_idx = list(alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).index).index(feature_to_change)
        criterion_range = self.msd_transformer.value_range[modified_criterion_idx]
        lower_bound = self.msd_transformer.lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.msd_transformer.objectives[modified_criterion_idx]

        # Positive and Negative Ideal Solution (utility space)
        PIS = weights
        NIS = np.zeros_like(performances_US)

        v_ij = performances_US * weights
        j = modified_criterion_idx

        # Calculate the sum of squared distances for the remaining (unmodified) criteria
        v_ij_excluding_j = np.delete(v_ij, j)
        PIS_excluding_j = np.delete(PIS, j)
        NIS_excluding_j = np.delete(NIS, j)
        k = (target_agg_value / (1 - target_agg_value)) ** 2
        p = k * np.sum((v_ij_excluding_j - PIS_excluding_j) ** 2) - np.sum((v_ij_excluding_j - NIS_excluding_j) ** 2)

        a = (1 - k) * (weights[j] / criterion_range) ** 2
        b = 2 * (weights[j] / criterion_range) * (v_ij[j] - NIS[j] - k * (v_ij[j] - PIS[j]))
        c = (v_ij[j] - NIS[j]) ** 2 - k * (v_ij[j] - PIS[j]) ** 2 - p

        solutions = TOPSISAggregationFunction.solve_quadratic_equation(a, b, c)  # solutions are performance modifications in CS !!!
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = solutions[0] + performances_CS[j]
            solution_2 = solutions[1] + performances_CS[j]

        # solution -- new performances in CS
        solution = TOPSISAggregationFunction.choose_appropriate_solution(solution_1, solution_2, lower_bound, upper_bound, objective)
        if solution is None:
            return None
        else:
            return solution - performances_CS[j]


    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):

      w = np.mean(self.msd_transformer.weights)
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
