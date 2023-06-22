class ImprovementActionsMixin:

    def improvement_mean(self, position, improvement, improvement_ratio):
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      m_boundary = np.mean(self.weights)
      if self.__defineAggregationFunction(m_boundary, alternative_to_improve["Std"]) < alternative_to_overcome["AggFn"]:
        print("It is impossible to improve with only mean")
      else:
        change = (m_boundary - alternative_to_improve["Mean"])/2
        actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
        while True:
          if actual_aggfn > alternative_to_overcome["AggFn"]:
            if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
              alternative_to_improve["Mean"] -= change
              change = change/2
              actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
            else:
              break
          else:
            alternative_to_improve["Mean"] += change
            change = change/2
            actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
        print("You should change mean by ", alternative_to_improve["Mean"] - self.data.loc[self.ranked_alternatives[position]]["Mean"])

    def improvement_std(self, position, improvement, improvement_ratio):
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      sd_boundary = np.mean(self.weights)/2
      if (self.agg_fn == "A") or (self.agg_fn == "R" and alternative_to_improve["Mean"]<sd_boundary):
        if self.__defineAggregationFunction(alternative_to_improve["Mean"], sd_boundary) < alternative_to_overcome["AggFn"]:
          print("It is impossible to improve with only standard deviation")
        else:
          change = (sd_boundary - alternative_to_improve["Std"])/2
          actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          while True:
            if actual_aggfn > alternative_to_overcome["AggFn"]:
              if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                alternative_to_improve["Std"] -= change
                change = change/2
                actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
              else:
                break
            else:
              alternative_to_improve["Std"] += change
              change = change/2
              actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          print("You should change standard deviation by ", alternative_to_improve["Std"] - self.data.loc[self.ranked_alternatives[position]]["Std"])
      else:
        if self.__defineAggregationFunction(alternative_to_improve["Mean"], 0) < alternative_to_overcome["AggFn"]:
          print("It is impossible to improve with only standard deviation")
        else:
          change = alternative_to_improve["Std"]/2
          actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          while True:
            if actual_aggfn > alternative_to_overcome["AggFn"]:
              if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                alternative_to_improve["Std"] += change
                change = change/2
                actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
              else:
                break
            else:
              alternative_to_improve["Std"] -= change
              change = change/2
              actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          print("You should change standard deviation by ", self.data.loc[self.ranked_alternatives[position]]["Std"] - alternative_to_improve["Std"])

    def improvement_full(self, position, improvement, improvement_ratio):
      alternative_to_improve = self.data.loc[self.ranked_alternatives[position]].copy()
      alternative_to_overcome = self.data.loc[self.ranked_alternatives[position - improvement]].copy()
      m_boundary = np.mean(self.weights)
      change_m = (m_boundary - alternative_to_improve["Mean"])/2
      change_sd = alternative_to_improve["Std"]/2
      actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
      while True:
        if actual_aggfn > alternative_to_overcome["AggFn"]:
          if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
            alternative_to_improve["Std"] += change_sd
            alternative_to_improve["Mean"] -= change_m
            change_sd = change_sd/2
            change_m = change_m/2
            actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
          else:
            break
        else:
          alternative_to_improve["Std"] -= change_sd
          alternative_to_improve["Mean"]+= change_m
          change_sd = change_sd/2
          change_m = change_m/2
          actual_aggfn = self.__defineAggregationFunction(alternative_to_improve["Mean"], alternative_to_improve["Std"])
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
        AggFn = self.__defineAggregationFunction(mean, std)
        
        if AggFn < alternative_to_overcome["AggFn"]:
          continue

        alternative_to_improve[i] = 0.5
        v = alternative_to_improve * w
        vw = (sum(v * w)/sum(w * w)) * w
        mean = np.sqrt(sum(vw*vw))/s
        std = np.sqrt(sum((v-vw)*(v-vw)))/s
        AggFn = self.__defineAggregationFunction(mean, std)
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
          AggFn = self.__defineAggregationFunction(mean, std)
        
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
