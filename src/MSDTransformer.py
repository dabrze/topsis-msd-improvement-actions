import math
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import plotly.graph_objects as go
from IPython.display import display
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


class MSDTransformer(TransformerMixin):

    def __init__(self, agg_fn, max_std_calculator="scip"):
        self.agg_fn = self.__check_agg_fn(agg_fn)
        self.max_std_calculator = self.__check_max_std_calculator(max_std_calculator)
        self.isFitted = False

    def fit(self, X, weights=None, objectives=None, expert_range=None):

        self.X = X
        self.n_alternatives = X.shape[0]
        self.n_criteria = X.shape[1]
        self.n = self.n_alternatives
        self.m = self.n_criteria

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

        self.__check_input()

        self.value_range = []
        self.lower_bounds = []
        for c in range(self.n_criteria):
            self.lower_bounds.append(self.expert_range[c][0])
            self.value_range.append(self.expert_range[c][1] - self.expert_range[c][0])

        self.weights = self.__normalize_weights(self.weights)
        self.X_new = self.__normalize_data(X.copy())
        self.__wmstd()
        self.X_new['AggFn'] = self.agg_fn.TOPSIS_calculation(np.mean(self.weights), self.X_new['Mean'], self.X_new['Std'])
        self.ranked_alternatives = self.__ranking()
        self.isFitted = True

        return self

    def fit_transform(self, X, weights=None, objectives=None, expert_range=None):
        self.fit(X, weights, objectives, expert_range)
        return self.X_new

    def transform(self, X):
        if not self.isFitted:
            raise Exception("fit is required before transform")

        self.__check_input_after_transform(X)
        X_transformed = self.__normalize_data(X.copy())
        w_means, w_stds = self.transform_US_to_wmsd(np.array(X_transformed))
        agg_values = self.agg_fn.TOPSIS_calculation(np.mean(self.weights), w_means, w_stds)
        X_transformed['Mean'] = w_means
        X_transformed['Std'] = w_stds
        X_transformed['AggFn'] = agg_values

        return X_transformed

    def transform_US_to_wmsd(self, X_US):
        # transform data from Utility Space to WMSD Space
        w = self.weights
        s = np.linalg.norm(w) / np.mean(w)
        v = X_US * w

        vw = (np.sum(v * w, axis=1) / np.sum(w ** 2)).reshape(-1, 1) @ w.reshape(1, -1)
        w_means = np.linalg.norm(vw, axis=1) / s
        w_stds = np.linalg.norm(v - vw, axis=1) / s
        return w_means, w_stds

    def inverse_transform(self, target_mean, target_std, std_type, sampling_density=None, epsilon=0.001):
        if sampling_density is None:
            sampling_density = math.ceil(5000000 ** (1 / self.n_criteria))
            # print("sampling_density", sampling_density)

        dims = [np.linspace(0, 1, sampling_density) for i in range(self.n_criteria)]
        grid = np.meshgrid(*dims)
        points = np.column_stack([xx.ravel() for xx in grid])
        # print(f"{len(points)} samples generated in total")
        w_means, w_stds = self.transform_US_to_wmsd(points)

        if std_type == "==":
            filtered_points = points[np.bitwise_and(abs(w_means - target_mean) < epsilon, abs(target_std - w_stds) < epsilon)]
        elif std_type == "<=":
            filtered_points = points[np.bitwise_and(abs(w_means - target_mean) < epsilon, w_stds <= target_std)]
        elif std_type == ">=":
            filtered_points = points[np.bitwise_and(abs(w_means - target_mean) < epsilon, w_stds >= target_std)]
        else:
            raise ValueError("Invalid value at `std_type`, should be one of the following strings '==', '<=', '>='")
        # TODO move validation of std_type before computationally expensive sampling and transformation

        # print(f"Result shape {filtered_points.shape}")
        return pd.DataFrame(filtered_points, columns=self.X.columns)

    def improvement(self, function_name, alternative_to_improve, alternative_to_overcome, improvement_ratio=0.000001, **kwargs):

        if type(alternative_to_improve) == int:
            alternative_to_improve = self.X_new.loc[self.ranked_alternatives[alternative_to_improve]].copy()
        elif type(alternative_to_improve) == str:
            alternative_to_improve = self.X_new.loc[alternative_to_improve].copy()

        if type(alternative_to_overcome) == int:
            alternative_to_overcome = self.X_new.loc[self.ranked_alternatives[alternative_to_overcome]].copy()
        elif type(alternative_to_overcome) == str:
            alternative_to_overcome = self.X_new.loc[alternative_to_overcome].copy()

        func = getattr(self.agg_fn, function_name)
        return func(alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs)

    def plot(self, heatmap_quality=500, show_names=False, plot_name=None):
        """
        Plots positions of alternatives in MSD space.
        """

        # for all possible mean and std count aggregation value and color it by it
        mean_ticks = np.linspace(0, 1, heatmap_quality + 1)
        std_ticks = np.linspace(0, 0.5, heatmap_quality // 2 + 1)
        grid = np.meshgrid(mean_ticks, std_ticks)
        w_means = grid[0].ravel()
        w_stds = grid[1].ravel()
        agg_values = self.agg_fn.TOPSIS_calculation(np.mean(self.weights), np.array(w_means), np.array(w_stds))
        if plot_name is None:
            plot_name = "Weights: " + ','.join([str(x) for x in self.weights])

        fig = go.Figure(data=go.Contour(
            x=w_means,
            y=w_stds,
            z=agg_values,
            zmin=0.0,
            zmax=1.0,
            colorscale='jet',
            contours_coloring='heatmap',
            line_width=0,
            colorbar=dict(
                title='Aggregation value',
                titleside='right',
                outlinewidth=1,
                title_font_size=22,
                tickfont_size=15

            ),
            hoverinfo='none'),
            layout=go.Layout(
                title=go.layout.Title(
                    text=plot_name,
                    font_size=30
                ),
                title_x=0.5,
                xaxis_range=[0.0, 1.0],
                yaxis_range=[0.0, 0.5]
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

        # calculate upper perimeter
        if len(set(self.weights)) == 1:
            def max_std(m, n):
                floor_mn = np.floor(m * n)
                nm = n * m
                value_under_sqrt = n * (floor_mn + (floor_mn - nm) ** 2) - nm ** 2
                return np.sqrt(value_under_sqrt) / n

            means = np.linspace(0, 1, 10000)
            perimeter = max_std(means, self.n_criteria)
        else:
            quality_exact = {
                2: 1000,
                3: 600,
                4: 400,
                5: 300,
                6: 200,
                7: 150,
                8: 125,
                9: 100,
            }
            means = np.linspace(0, np.mean(self.weights), quality_exact.get(self.n_criteria, 50))
            perimeter = [self.max_std_calculator(mean, self.weights) for mean in means]

        # draw upper perimeter
        fig.add_trace(go.Scatter(
            x=means,
            y=perimeter,
            mode='lines',
            showlegend=False,
            hoverinfo='none',
            line_color='black'
        ))

        # fill between the line and the std = 0.5
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0.5, 0.5],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 1)',
            showlegend=False,
            hoverinfo='none',
            line_color='white'
        ))

        # fill from the end of the graph to mean = 1
        fig.add_trace(go.Scatter(
            x=[max(means), max(means), 1],
            y=[0, 0.5, 0.5],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(255, 255, 255, 1)',
            showlegend=False,
            hoverinfo='none',
            line_color='white'
        ))

        self.plot_background = go.Figure(fig)
        values = self.X_new['AggFn'].to_list()

        ### plot the ranked data
        custom = []
        for i in self.X_new.index.values:
            custom.append(1 + self.ranked_alternatives.index(i))

        fig.add_trace(go.Scatter(
            x=self.X_new['Mean'].tolist(),
            y=self.X_new['Std'].tolist(),
            showlegend=False,
            mode='markers',
            marker=dict(
                color='black',
                size=10
            ),
            customdata=np.stack((custom, values), axis=1),
            text=self.X_new.index.values,
            hovertemplate='<b>ID</b>: %{text}<br>' +
                          '<b>Rank</b>: %{customdata[0]:f}<br>' +
                          '<b>AggFn</b>: %{customdata[1]:f}<br>' +
                          '<extra></extra>'
        ))
        ### add names
        if show_names == True:
            for i, label in enumerate(self.X_new.index.values):
                fig.add_annotation(
                    x=self.X_new['Mean'].tolist()[i] + 0.01,
                    y=self.X_new['Std'].tolist()[i] + 0.01,
                    text=label,
                    showarrow=False,
                    font=dict(size=12)
                )
        return fig

    def update_for_plot(self, id, changes, change_number):
        if 'Mean' in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc['NEW ' + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc['NEW ' + id, "Mean"] += changes["Mean"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(np.mean(self.weights), self.X_newPoint.loc['NEW ' + id, "Mean"], self.X_newPoint.loc['NEW ' + id, "Std"])
            self.X_newPoint.loc['NEW ' + id, "AggFn"] = agg_value
        elif 'Std' in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc['NEW ' + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc['NEW ' + id, "Std"] += changes["Std"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(np.mean(self.weights), self.X_newPoint.loc['NEW ' + id, "Mean"], self.X_newPoint.loc['NEW ' + id, "Std"])
            self.X_newPoint.loc['NEW ' + id, "AggFn"] = agg_value
        else:
            row_to_add = changes.iloc[change_number]
            result = self.X.loc[id].add(row_to_add, fill_value=0)
            new_row = pd.Series(result, name='NEW ' + id)
            self.X_newPoint = self.X.copy()
            self.X_newPoint = self.X_newPoint.append(new_row)
            self.X_newPoint = self.__normalize_data(self.X_newPoint)
            w_means, w_stds = self.transform_US_to_wmsd(np.array(self.X_newPoint))
            agg_values = self.agg_fn.TOPSIS_calculation(np.mean(self.weights), w_means, w_stds)
            self.X_newPoint['Mean'] = w_means
            self.X_newPoint['Std'] = w_stds
            self.X_newPoint['AggFn'] = agg_values
        return self.X_newPoint

    def plot2(self, id, changes, show_names=False, change_number=0):
        self.update_for_plot(id, changes, change_number)
        old_rank = self.X_new.sort_values(by='AggFn', ascending=False).index.get_loc(id) + 1
        old_value = self.X_new.loc[id, 'AggFn']
        fig = self.plot_background

        ### add old point
        fig.add_trace(go.Scatter(
            x=[self.X_new.loc[id, 'Mean']],
            y=[self.X_new.loc[id, 'Std']],
            showlegend=False,
            mode='markers',
            marker=dict(
                color='white',
                size=10
            ),
            customdata=np.stack(([old_rank], [old_value]), axis=1),
            text=['OLD ' + id],
            hovertemplate='<b>ID</b>: %{text}<br>' +
                          '<b>Old Rank</b>: %{customdata[0]:f}<br>' +
                          '<b>New Rank</b>: -<br>' +
                          '<b>AggFn</b>: %{customdata[1]:f}<br>' +
                          '<extra></extra>'
        ))
        ### add name for old point
        if show_names == True:
            fig.add_annotation(
                x=self.X_new.loc[id, 'Mean'] + 0.01,
                y=self.X_new.loc[id, 'Std'] + 0.01,
                text='OLD ' + id,
                showarrow=False,
                font=dict(size=12)
            )

        self.X_newPoint = self.X_newPoint.drop(index=(id))
        self.X_newPoint = self.X_newPoint.sort_values(by='AggFn', ascending=False)
        new_rank = self.X_newPoint.index.get_loc('NEW ' + id) + 1
        new_value = self.X_newPoint.loc['NEW ' + id, 'AggFn']

        numbers = [i for i in range(1, self.n + 1)]
        custom0 = numbers.copy()
        custom0.remove(old_rank)
        custom1 = numbers.copy()
        custom1.remove(new_rank)

        ### add new point
        fig.add_trace(go.Scatter(
            x=[self.X_newPoint.loc['NEW ' + id, 'Mean']],
            y=[self.X_newPoint.loc['NEW ' + id, 'Std']],
            showlegend=False,
            mode='markers',
            marker=dict(
                color='white',
                size=10
            ),
            customdata=np.stack(([new_rank], [new_value]), axis=1),
            text=['NEW ' + id],
            hovertemplate='<b>ID</b>: %{text}<br>' +
                          '<b>Old Rank</b>: -<br>' +
                          '<b>New Rank</b>: %{customdata[0]:f}<br>' +
                          '<b>AggFn</b>: %{customdata[1]:f}<br>' +
                          '<extra></extra>'
        ))
        ### add names for new point and other points
        if show_names == True:
            for i, label in enumerate(self.X_newPoint.index.values):
                fig.add_annotation(
                    x=self.X_newPoint['Mean'].tolist()[i] + 0.01,
                    y=self.X_newPoint['Std'].tolist()[i] + 0.01,
                    text=label,
                    showarrow=False,
                    font=dict(size=12)
                )
        ### add line between old point and new point
        fig.add_shape(
            type='line',
            x0=self.X_new.loc[id, 'Mean'],
            y0=self.X_new.loc[id, 'Std'],
            x1=self.X_newPoint.loc['NEW ' + id, 'Mean'],
            y1=self.X_newPoint.loc['NEW ' + id, 'Std'],
            line=dict(color='white', width=2),
        )
        self.X_newPoint = self.X_newPoint.drop(index=('NEW ' + id))
        values = self.X_newPoint['AggFn'].to_list()

        ### add other points
        fig.add_trace(go.Scatter(
            x=self.X_newPoint['Mean'].tolist(),
            y=self.X_newPoint['Std'].tolist(),
            showlegend=False,
            mode='markers',
            marker=dict(
                color='black',
                size=10
            ),
            customdata=np.stack((custom0, custom1, values), axis=1),
            text=self.X_newPoint.index.values,
            hovertemplate='<b>ID</b>: %{text}<br>' +
                          '<b>Old Rank</b>: %{customdata[0]:f}<br>' +
                          '<b>New Rank</b>: %{customdata[1]:f}<br>' +
                          '<b>AggFn</b>: %{customdata[2]:f}<br>' +
                          '<extra></extra>'
        ))
        return fig

    def show_ranking(self, mode=None, first=1, last=None):

        if last is None:
            last = len(self.X_new.index)

        self.__check_show_ranking(first, last)

        ranking = self.X_new
        ranking = ranking.assign(Rank=None)
        columns = ranking.columns.tolist()
        columns = columns[-1:] + columns[:-1]
        ranking = ranking[columns]

        alternative_names = ranking.index.tolist()
        for alternative in alternative_names:
            ranking['Rank'][alternative] = self.ranked_alternatives.index(alternative) + 1

        ranking = ranking.sort_values(by=['Rank'])
        # ranking = ranking.loc[max(first-1, 0):last]
        ranking = ranking[(first - 1):last]

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

    def __check_max_std_calculator(self, max_std_calculator):
        if isinstance(max_std_calculator, str):
            if max_std_calculator == "scip":
                from utils.max_std_calculator_scip import max_std_scip
                return max_std_scip
            elif max_std_calculator == "gurobi":
                from utils.max_std_calculator_gurobi import max_std_gurobi
                return max_std_gurobi
            else:
                raise ValueError("Invalid value at 'agg_fn': must be string (gurobi or scip) or function.")
        elif callable(max_std_calculator):
            return max_std_calculator
        else:
            raise ValueError("Invalid value at 'agg_fn': must be string (scip or gurobi) or function.")

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
            return self.__dict_to_list(weights)

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
            return self.__dict_to_list(objectives)
        elif objectives is None:
            return np.repeat('max', self.m)
        else:
            raise ValueError("Invalid value at 'objectives': must be a list or a string (gain, g, cost, c, min or max) or a dictionary")

    def __check_expert_range(self, expert_range):
        if isinstance(expert_range, dict):
            expert_range = self.__dict_to_list(expert_range)

        if isinstance(expert_range, list):

            if all(isinstance(e, list) for e in expert_range):
                return expert_range

            elif all(isinstance(e, (int, float, np.float64)) for e in expert_range):
                expert_range = [expert_range]
                numpy_expert_range = np.repeat(expert_range, self.m, axis=0)
                return numpy_expert_range.tolist()

            else:
                raise ValueError("Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary")

        elif expert_range is None:
            lower_bounds = self.X.min()
            upper_bounds = self.X.max()
            expert_range = [lower_bounds, upper_bounds]
            numpy_expert_range = np.array(expert_range).T
            return numpy_expert_range.tolist()

        else:
            raise ValueError("Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary")

    def __check_input(self):

        if self.X.isnull().values.any():
            raise ValueError("Dataframe must not contain any none/nan values, but found at least one")

        if len(self.weights) != self.m:
            raise ValueError("Invalid value 'weights'.")

        if not all(type(item) in [int, float, np.float64] for item in self.weights):
            raise ValueError("Invalid value 'weights'. Expected numerical value (int or float).")

        if not all(item >= 0 for item in self.weights):
            raise ValueError("Invalid value 'weights'. Expected value must be non-negative.")

        if not any(item > 0 for item in self.weights):
            raise ValueError("Invalid value 'weights'. At least one weight must be positive.")

        if len(self.objectives) != self.m:
            raise ValueError("Invalid value 'objectives'.")

        if not all(item in ["min", "max"] for item in self.objectives):
            raise ValueError(
                "Invalid value at 'objectives'. Use 'min', 'max', 'gain', 'cost', 'g' or 'c'.")

        if len(self.expert_range) != len(self.objectives):
            raise ValueError(
                "Invalid value at 'expert_range'. Length of should be equal to number of criteria.")

        for col in self.expert_range:
            if len(col) != 2:
                raise ValueError(
                    "Invalid value at 'expert_range'. Every criterion has to have minimal and maximal value.")
            if not all(type(item) in [int, float] for item in col):
                raise ValueError(
                    "Invalid value at 'expert_range'. Expected numerical value (int or float).")
            if col[0] > col[1]:
                raise ValueError("Invalid value at 'expert_range'. Minimal value  is bigger then maximal value.")

        lower_bound = np.array(self.X.min()).tolist()
        upper_bound = np.array(self.X.max()).tolist()

        for val, mini, maxi in zip(self.expert_range, lower_bound, upper_bound):
            if not (val[0] <= mini and val[1] >= maxi):
                raise ValueError("Invalid value at 'expert_range'. All values from original data must be in a range of expert_range.")

    def __check_input_after_transform(self, X):
        n = X.shape[0]
        m = X.shape[1]

        if X.isnull().values.any():
            raise ValueError("Dataframe must not contain any none/nan values, but found at least one")

        if self.n_criteria != m:
            raise ValueError("Invalid number of columns. Number of criteria must be the same as in previous dataframe.")

        if not all(X.columns.values == self.X.columns.values):
            raise ValueError("New dataset must have the same columns as the dataset used to fit MSDTransformer")

        lower_bound = np.array(X.min()).tolist()
        upper_bound = np.array(X.max()).tolist()

        for val, mini, maxi in zip(self.expert_range, lower_bound, upper_bound):
            if not (val[0] <= mini and val[1] >= maxi):
                raise ValueError("Invalid value at 'expert_range'. All values from original data must be in a range of expert_range.")

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

    def __normalize_data(self, data):
        """normalize given data using either given expert range or min/max
        uses the min-max normalization with minimum and maximum taken from expert ranges if given
        Parameters
        ----------
        data : dataframe
            data to be normalized
        """
        c = 0
        for col in data.columns:
            data[col] = (data[col] - self.expert_range[c][0]) / \
                        (self.expert_range[c][1] - self.expert_range[c][0])
            c += 1

        for i in range(self.m):
            if self.objectives[i] == 'min':
                data[data.columns[i]] = 1 - data[data.columns[i]]

        return data

    def __normalize_weights(self, weights):
        """normalize weights
        result are weights not greater than 1 but not 0 if not present previously
        Parameters
        ----------
        weights : np.array
            weights to be normalized
        """
        weights = np.array([float(i) / max(weights) for i in weights])
        return weights

    def __wmstd(self):

        w = self.weights
        s = np.sqrt(sum(w * w)) / np.mean(w)
        wm = []
        wsd = []
        for index, row in self.X_new.iterrows():
            v = row * w
            vw = (sum(v * w) / sum(w * w)) * w
            wm.append(np.sqrt(sum(vw * vw)) / s)
            wsd.append(np.sqrt(sum((v - vw) * (v - vw))) / s)

        self.X_new['Mean'] = wm
        self.X_new['Std'] = wsd

    def __ranking(self):
        """creates a ranking from the data based on topsis value column"""
        data__ = self.X_new.copy()
        data__ = data__.sort_values(by='AggFn', ascending=False)
        arranged = data__.index.tolist()
        return arranged

    def __dict_to_list(self, dictionary):
        new_list = []

        for col_name in self.X.columns:
            new_list.append(dictionary[col_name])

        return new_list


class TOPSISAggregationFunction(ABC):
    def __init__(self, msd_transformer):
        self.msd_transformer = msd_transformer

    @abstractmethod
    def TOPSIS_calculation(self, w, wm, wsd):
        pass

    @abstractmethod
    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change, **kwargs):
        pass

    def improvement_mean(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):
        if alternative_to_improve["AggFn"] >= alternative_to_overcome["AggFn"]:
            raise ValueError("Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'")

        w = np.mean(self.msd_transformer.weights)
        m_start = alternative_to_improve["Mean"]
        m_boundary = w
        if self.TOPSIS_calculation(w, m_boundary, alternative_to_improve["Std"]) < alternative_to_overcome["AggFn"]:
            return None
        else:
            change = (m_boundary - alternative_to_improve["Mean"]) / 2
            actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
            while True:
                if actual_aggfn >= alternative_to_overcome["AggFn"]:
                    if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                        alternative_to_improve["Mean"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                    else:
                        break
                else:
                    alternative_to_improve["Mean"] += change
                    actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                    if actual_aggfn >= alternative_to_overcome["AggFn"]:
                        change = change / 2
            if alternative_to_improve["Std"] <= self.msd_transformer.max_std_calculator(alternative_to_improve["Mean"], self.msd_transformer.weights):
                return pd.DataFrame([alternative_to_improve["Mean"] - m_start], columns=["Mean"])
            else:
                while alternative_to_improve["Mean"] <= 1:
                    if alternative_to_improve["Std"] <= self.msd_transformer.max_std_calculator(alternative_to_improve["Mean"], self.msd_transformer.weights):
                        return pd.DataFrame([alternative_to_improve["Mean"] - m_start], columns=["Mean"])
                    alternative_to_improve["Mean"] += improvement_ratio
                return None

    def __check_boundary_values(self, alternative_to_improve, features_to_change, boundary_values):
        if boundary_values is None:
            boundary_values = np.ones(len(features_to_change))
        else:
            if len(features_to_change) != len(boundary_values):
                raise ValueError("Invalid value at 'boundary_values': must be same length as 'features_to_change'")
            for i in range(len(features_to_change)):
                col = self.msd_transformer.X_new.columns.get_loc(features_to_change[i])
                if boundary_values[i] < self.msd_transformer.expert_range[col][0] or boundary_values[i] > self.msd_transformer.expert_range[col][1]:
                    raise ValueError("Invalid value at 'boundary_values': must be between defined 'expert_range'")
                else:
                    boundary_values[i] = (boundary_values[i] - self.msd_transformer.expert_range[col][0]) / (self.msd_transformer.expert_range[col][1] - self.msd_transformer.expert_range[col][0])
                    if self.msd_transformer.objectives[col] == "min":
                        boundary_values[i] = 1 - boundary_values[i]
                    if alternative_to_improve[features_to_change[i]] > boundary_values[i]:
                        raise ValueError("Invalid value at 'boundary_values': must be better or equal to improving alternative values")
        return np.array(boundary_values)

    def improvement_features(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, features_to_change, boundary_values=None, **kwargs):
        if alternative_to_improve["AggFn"] >= alternative_to_overcome["AggFn"]:
            raise ValueError("Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'")
        boundary_values = self.__check_boundary_values(alternative_to_improve, features_to_change, boundary_values)

        AggFn = alternative_to_improve["AggFn"]
        alternative_to_improve = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"])
        improvement_start = alternative_to_improve.copy()
        feature_pointer = 0
        w = self.msd_transformer.weights
        value_range = self.msd_transformer.value_range
        objectives = self.msd_transformer.objectives

        is_improvement_satisfactory = False

        s = np.sqrt(sum(w * w)) / np.mean(w)
        for i, k in zip(features_to_change, boundary_values):
            alternative_to_improve[i] = k
            mean, std = self.msd_transformer.transform_US_to_wmsd([alternative_to_improve])
            AggFn = self.TOPSIS_calculation(np.mean(w), mean, std)

            if AggFn < alternative_to_overcome["AggFn"]:
                continue

            alternative_to_improve[i] = 0.5 * k
            mean, std = self.msd_transformer.transform_US_to_wmsd([alternative_to_improve])
            AggFn = self.TOPSIS_calculation(np.mean(w), mean, std)
            change_ratio = 0.25 * k
            while True:
                if AggFn < alternative_to_overcome["AggFn"]:
                    alternative_to_improve[i] += change_ratio
                elif AggFn - alternative_to_overcome["AggFn"] > improvement_ratio:
                    alternative_to_improve[i] -= change_ratio
                else:
                    is_improvement_satisfactory = True
                    break
                change_ratio = change_ratio / 2
                mean, std = self.msd_transformer.transform_US_to_wmsd([alternative_to_improve])
                AggFn = self.TOPSIS_calculation(np.mean(w), mean, std)

            if is_improvement_satisfactory:
                alternative_to_improve -= improvement_start
                for j in range(len(alternative_to_improve)):
                    if alternative_to_improve[j] == 0:
                        continue
                    elif objectives[j] == "max":
                        alternative_to_improve[j] = value_range[j] * alternative_to_improve[j]
                    else:
                        alternative_to_improve[j] = -value_range[j] * alternative_to_improve[j]
                result_df = alternative_to_improve.to_frame().transpose()
                result_df = result_df.reset_index(drop=True)
                return result_df
        else:
            return None

    def improvement_genetic(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, features_to_change, boundary_values=None, allow_deterioration=False, popsize=None,
                            n_generations=200):
        boundary_values = self.__check_boundary_values(alternative_to_improve, features_to_change, boundary_values)

        current_performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        modified_criteria_subset = [x in features_to_change for x in self.msd_transformer.X.columns.tolist()]

        max_possible_improved = current_performances_US.copy()
        max_possible_improved[modified_criteria_subset] = boundary_values
        w_means, w_stds = self.msd_transformer.transform_US_to_wmsd(np.array([max_possible_improved]))
        max_possible_agg_value = self.TOPSIS_calculation(np.mean(self.msd_transformer.weights), w_means, w_stds).item()
        if max_possible_agg_value < alternative_to_overcome["AggFn"]:
            # print(f"Not possible to achieve target {alternative_to_overcome['AggFn']} with specified features and boundary_values. Max possible agg value is {max_possible_agg_value}")
            return None

        problem = PostFactumTopsisPymoo(
            topsis_model=self.msd_transformer,
            modified_criteria_subset=modified_criteria_subset,
            current_performances=current_performances_US,
            target_agg_value=alternative_to_overcome["AggFn"],
            upper_bounds=boundary_values,
            allow_deterioration=allow_deterioration)

        if popsize is None:
            popsize_by_n_objectives = {
                2: 150,
                3: 500,
                4: 1000
            }
            popsize = popsize_by_n_objectives.get(len(features_to_change), 2000)

        algorithm = NSGA2(pop_size=popsize,
                          crossover=SBX(eta=15, prob=0.9),
                          mutation=PM(eta=20),
                          save_history=False)

        res = minimize(problem, algorithm,
                       termination=('n_gen', n_generations),
                       seed=42, verbose=False)

        if res.F is not None:
            improvement_actions = np.zeros(shape=(len(res.F), len(current_performances_US)))
            improvement_actions[:, modified_criteria_subset] = res.F - current_performances_US[modified_criteria_subset]
            improvement_actions *= np.array(self.msd_transformer.value_range)
            improvement_actions[:, np.array(self.msd_transformer.objectives) == "min"] *= -1
            return pd.DataFrame(sorted(improvement_actions.tolist(), key=lambda x: x[0]), columns=self.msd_transformer.X.columns)
        else:
            return None

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


class PostFactumTopsisPymoo(Problem):
    def __init__(self, topsis_model, modified_criteria_subset, current_performances, target_agg_value, upper_bounds, allow_deterioration=False):
        n_criteria = np.array(modified_criteria_subset).astype(bool).sum()
        super().__init__(n_var=n_criteria, n_obj=n_criteria, n_ieq_constr=1, vtype=float)

        self.topsis_model = topsis_model
        self.mean_of_weights = np.mean(self.topsis_model.weights)
        self.modified_criteria_subset = np.array(modified_criteria_subset).astype(bool)
        self.current_performances = current_performances.copy()
        self.target_agg_value = target_agg_value

        # Lower and upper bounds in Utility Space
        self.xl = np.zeros(n_criteria) if allow_deterioration else self.current_performances[self.modified_criteria_subset]
        self.xu = upper_bounds

    def _evaluate(self, x, out, *args, **kwargs):
        # In Utility Space variables and objectives are the same values
        out["F"] = x.copy()  # this copy might be redundant

        # Topsis target constraint
        modified_performances = np.repeat([self.current_performances], repeats=len(x), axis=0)
        modified_performances[:, self.modified_criteria_subset] = x.copy()  # this copy might be redundant
        w_means, w_stds = self.topsis_model.transform_US_to_wmsd(modified_performances)
        agg_values = self.topsis_model.agg_fn.TOPSIS_calculation(self.mean_of_weights, w_means, w_stds)
        g1 = self.target_agg_value - agg_values  # In Pymoo positive values indicate constraint violation
        out["G"] = np.array([g1])


class ATOPSIS(TOPSISAggregationFunction):
    def __init__(self, msd_transformer):
        super().__init__(msd_transformer)

    def TOPSIS_calculation(self, w, wm, wsd):

        return np.sqrt(wm * wm + wsd * wsd) / w

    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change, **kwargs):
        """ Exact algorithm dedicated to the aggregation `A` for achieving the target by modifying the performance on a single criterion. """
        performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        performances_CS = performances_US * self.msd_transformer.value_range + self.msd_transformer.lower_bounds
        weights = self.msd_transformer.weights
        target_agg_value = (alternative_to_overcome["AggFn"] + improvement_ratio / 2) * np.linalg.norm(weights)

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
                feature_modification = solution - performances_CS[j]
                modification_vector = np.zeros_like(performances_US)
                modification_vector[modified_criterion_idx] = feature_modification
                result_df = pd.DataFrame([modification_vector], columns=self.msd_transformer.X.columns)
                return result_df

    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):
        if alternative_to_improve["AggFn"] >= alternative_to_overcome["AggFn"]:
            raise ValueError("Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'")

        w = np.mean(self.msd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        sd_boundary = self.msd_transformer.max_std_calculator(alternative_to_improve["Mean"], self.msd_transformer.weights)
        if self.TOPSIS_calculation(w, alternative_to_improve["Mean"], sd_boundary) < alternative_to_overcome["AggFn"]:
            return None
        else:
            change = (sd_boundary - alternative_to_improve["Std"]) / 2
            actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
            while True:
                if actual_aggfn > alternative_to_overcome["AggFn"]:
                    if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                        alternative_to_improve["Std"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                    else:
                        break
                else:
                    alternative_to_improve["Std"] += change
                    change = change / 2
                    actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
            return pd.DataFrame([alternative_to_improve["Std"] - std_start], columns=["Std"])


class ITOPSIS(TOPSISAggregationFunction):
    def __init__(self, msd_transformer):
        super().__init__(msd_transformer)

    def TOPSIS_calculation(self, w, wm, wsd):
        return 1 - np.sqrt((w - wm) * (w - wm) + wsd * wsd) / w

    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change, **kwargs):
        """ Exact algorithm dedicated to the aggregation `I` for achieving the target by modifying the performance on a single criterion. """
        performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        performances_CS = performances_US * self.msd_transformer.value_range + self.msd_transformer.lower_bounds
        weights = self.msd_transformer.weights
        target_agg_value = (1 - (alternative_to_overcome["AggFn"] + improvement_ratio / 2)) * np.linalg.norm(weights)

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
                feature_modification = solution - performances_CS[j]
                modification_vector = np.zeros_like(performances_US)
                modification_vector[modified_criterion_idx] = feature_modification
                result_df = pd.DataFrame([modification_vector], columns=self.msd_transformer.X.columns)
                return result_df

    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):
        if alternative_to_improve["AggFn"] >= alternative_to_overcome["AggFn"]:
            raise ValueError("Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'")

        w = np.mean(self.msd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        sd_boundary = self.msd_transformer.max_std_calculator(alternative_to_improve["Mean"], self.msd_transformer.weights)
        if self.TOPSIS_calculation(w, alternative_to_improve["Mean"], 0) < alternative_to_overcome["AggFn"]:
            return None
        else:
            change = alternative_to_improve["Std"] / 2
            actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
            while True:
                if actual_aggfn > alternative_to_overcome["AggFn"]:
                    if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                        alternative_to_improve["Std"] += change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                    else:
                        break
                else:
                    alternative_to_improve["Std"] -= change
                    change = change / 2
                    actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
            return pd.DataFrame([alternative_to_improve["Std"] - std_start], columns=["Std"])


class RTOPSIS(TOPSISAggregationFunction):
    def __init__(self, msd_transformer):
        super().__init__(msd_transformer)

    def TOPSIS_calculation(self, w, wm, wsd):
        return np.sqrt(wm * wm + wsd * wsd) / (np.sqrt(wm * wm + wsd * wsd) + np.sqrt((w - wm) * (w - wm) + wsd * wsd))

    def improvement_single_feature(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, feature_to_change, **kwargs):
        """ Exact algorithm dedicated to the aggregation `R` for achieving the target by modifying the performance on a single criterion. """
        performances_US = alternative_to_improve.drop(labels=["Mean", "Std", "AggFn"]).to_numpy().copy()
        performances_CS = performances_US * self.msd_transformer.value_range + self.msd_transformer.lower_bounds
        weights = self.msd_transformer.weights
        target_agg_value = alternative_to_overcome["AggFn"] + improvement_ratio / 2

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
            feature_modification = solution - performances_CS[j]
            modification_vector = np.zeros_like(performances_US)
            modification_vector[modified_criterion_idx] = feature_modification
            result_df = pd.DataFrame([modification_vector], columns=self.msd_transformer.X.columns)
            return result_df

    def improvement_std(self, alternative_to_improve, alternative_to_overcome, improvement_ratio, **kwargs):
        if alternative_to_improve["AggFn"] >= alternative_to_overcome["AggFn"]:
            raise ValueError("Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'")

        w = np.mean(self.msd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        sd_boundary = self.msd_transformer.max_std_calculator(alternative_to_improve["Mean"], self.msd_transformer.weights)
        if (alternative_to_improve["Mean"] < w / 2):
            if self.TOPSIS_calculation(w, alternative_to_improve["Mean"], sd_boundary) < alternative_to_overcome["AggFn"]:
                return None
            else:
                change = (sd_boundary - alternative_to_improve["Std"]) / 2
                actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                while True:
                    if actual_aggfn > alternative_to_overcome["AggFn"]:
                        if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                            alternative_to_improve["Std"] -= change
                            change = change / 2
                            actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                        else:
                            break
                    else:
                        alternative_to_improve["Std"] += change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                return pd.DataFrame([alternative_to_improve["Std"] - std_start], columns=["Improvement rate"], index=["Std"])
        else:
            if self.TOPSIS_calculation(w, alternative_to_improve["Mean"], 0) < alternative_to_overcome["AggFn"]:
                return None
            else:
                change = alternative_to_improve["Std"] / 2
                actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                while True:
                    if actual_aggfn > alternative_to_overcome["AggFn"]:
                        if actual_aggfn - alternative_to_overcome["AggFn"] > improvement_ratio:
                            alternative_to_improve["Std"] += change
                            change = change / 2
                            actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                        else:
                            break
                    else:
                        alternative_to_improve["Std"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(w, alternative_to_improve["Mean"], alternative_to_improve["Std"])
                return pd.DataFrame([alternative_to_improve["Std"] - std_start], columns=["Std"])
