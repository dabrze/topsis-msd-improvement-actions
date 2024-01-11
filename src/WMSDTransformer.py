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
from sklearn.cluster import AgglomerativeClustering


class WMSDTransformer(TransformerMixin):
    """
    A class used to calculate TOPSIS ranking,
    plot positions of alternatives in WMSD space,
    perform improvement actions on selected alternative.

    X : data-frame
        Pandas data-frame provided by the user.
    X_new : data-frame
        Pandas data-frame, normalized X.
    data : data-frame
        A copy of self.X, on which all calculations are performed.
    n : int
        Number of data-frame's columns
    m : int
        Number of data-frame's rows
    weights : np.array of float
        Array containing normalized weights.
    objectives : np.array of str
        Numpy array informing which criteria are cost type
        and which are gain type.
    expert_range : 2D list of floats
        2D list containing normalized expert range.
    """

    def __init__(self, agg_fn, max_std_calculator="scip"):
        self.agg_fn = self.__check_agg_fn(agg_fn)
        self.max_std_calculator = self.__check_max_std_calculator(max_std_calculator)
        self._isFitted = False

    def fit(self, X, weights=None, objectives=None, expert_range=None):

        """Checks input data and normalizes it.
        Parameters
        ----------
        
        X : data-frame
            Pandas data-frame provided by the user.
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
        expert_range : 2D list or dictionary, optional
            For each criterion must be provided minimal and maximal value.
            All criteria must fit in range [minimal, maximal]
            (default: 2D list of minimal and maximal values among provided criteria)
        """

        self.X = X
        self.n = X.shape[0]  # n_alternatives
        self.m = X.shape[1]  # n_criteria
        #print(weights)

        self._original_weights = self.__check_weights(weights)
        self.weights = self._original_weights.copy()

        self.objectives = self.__check_objectives(objectives)

        self.objectives = list(map(lambda x: x.replace("gain", "max"), self.objectives))
        self.objectives = list(map(lambda x: x.replace("g", "max"), self.objectives))
        self.objectives = list(map(lambda x: x.replace("cost", "min"), self.objectives))
        self.objectives = list(map(lambda x: x.replace("c", "min"), self.objectives))

        self.expert_range = self.__check_expert_range(expert_range)

        self._ranked_alternatives = []

        self.__check_input()

        self._value_range = []
        self._lower_bounds = []
        for c in range(self.m):
            self._lower_bounds.append(self.expert_range[c][0])
            self._value_range.append(self.expert_range[c][1] - self.expert_range[c][0])

        self.weights = self.__normalize_weights(self.weights)
        #print(self.weights)
        self.X_new = self.__normalize_data(X.copy())
        self.__wmstd()
        self.X_new[str(self.agg_fn.letter)] = self.agg_fn.TOPSIS_calculation(
            np.mean(self.weights), self.X_new["Mean"], self.X_new["Std"]
        )
        self._ranked_alternatives = self.__ranking()
        self._isFitted = True

        return self

    def transform(self, X):
        """Transform data from data-frame X to WMSD space.
        Parameters
        ----------

        X : pandas data-frame
            Data frame that contains data to be transformed.
        Returns
        -------
        Pandas data-frame.
        """
        if not self._isFitted:
            raise Exception("fit is required before transform")

        self.__check_input_after_transform(X)
        X_transformed = self.__normalize_data(X.copy())
        w_means, w_stds = self.transform_US_to_wmsd(np.array(X_transformed))
        agg_values = self.agg_fn.TOPSIS_calculation(
            np.mean(self.weights), w_means, w_stds
        )
        X_transformed["Mean"] = w_means
        X_transformed["Std"] = w_stds
        X_transformed[str(self.agg_fn.letter)] = agg_values

        return X_transformed
    
    def fit_transform(self, X, weights=None, objectives=None, expert_range=None):
        """Runs fit() method.
        Parameters
        ----------
        
        X : data-frame
            Pandas data-frame provided by the user.
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
        expert_range : 2D list or dictionary, optional
            For each criterion must be provided minimal and maximal value.
            All criteria must fit in range [minimal, maximal]
            (default: 2D list of minimal and maximal values among provided criteria)
        """
        self.fit(X, weights, objectives, expert_range)
        return self.X_new

    def transform_US_to_wmsd(self, X_US):

        # transform data from Utility Space to WMSD Space
        w = self.weights
        s = np.linalg.norm(w) / np.mean(w)
        v = X_US * w

        vw = (np.sum(v * w, axis=1) / np.sum(w**2)).reshape(-1, 1) @ w.reshape(1, -1)
        w_means = np.linalg.norm(vw, axis=1) / s
        w_stds = np.linalg.norm(v - vw, axis=1) / s
        return w_means, w_stds

    def inverse_transform(
        self, target_mean, target_std, std_type, sampling_density=None, epsilon=0.01
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if sampling_density is None:
            sampling_density = math.ceil(5000000 ** (1 / self.m))
            # print("sampling_density", sampling_density)

        dims = [np.linspace(0, 1, sampling_density) for i in range(self.m)]
        grid = np.meshgrid(*dims)
        points = np.column_stack([xx.ravel() for xx in grid])
        # print(f"{len(points)} samples generated in total")
        w_means, w_stds = self.transform_US_to_wmsd(points)

        if std_type == "==":
            filtered_points = points[
                np.bitwise_and(
                    abs(w_means - target_mean) < epsilon,
                    abs(target_std - w_stds) < epsilon,
                )
            ]
        elif std_type == "<=":
            filtered_points = points[
                np.bitwise_and(
                    abs(w_means - target_mean) < epsilon, w_stds <= target_std
                )
            ]
        elif std_type == ">=":
            filtered_points = points[
                np.bitwise_and(
                    abs(w_means - target_mean) < epsilon, w_stds >= target_std
                )
            ]
        else:
            raise ValueError(
                "Invalid value at `std_type`, should be one of the following strings '==', '<=', '>='"
            )
        # TODO move validation of std_type before computationally expensive sampling and transformation

        # print(f"Result shape {filtered_points.shape}")
        return pd.DataFrame(filtered_points, columns=self.X.columns)

    def plot(self, heatmap_quality=500, show_names=False, plot_name=None, color='jet'):

        """Plots positions of alternatives in WMSD space.
        Parameters
        ----------
        heatmap_quality : int, optional
            Integer value of the precision of the heatmap. The higher, the heatmap more fills the plot, but the plot generates longer.
            (default: 500)
        show_names : bool, optional
            Boolean value, if true, then points labels are showed on a plot.
            (default: False)
        plot_name : str, optional
            String that contains a title of a plot. If None, then in place of title, weights of criteria will be shown.
            (default: None)
        color : str, optional
            String that contains a name of a color-scale in which the plot will be presented.
            (default: 'jet')
        Returns
        -------
        Plot as an plotly figure.
        """

        # for all possible mean and std count aggregation value and color it by it
        mean_ticks = np.linspace(0, 1, heatmap_quality + 1)
        std_ticks = np.linspace(0, 0.5, heatmap_quality // 2 + 1)
        grid = np.meshgrid(mean_ticks, std_ticks)
        w_means = grid[0].ravel()
        w_stds = grid[1].ravel()
        agg_values = self.agg_fn.TOPSIS_calculation(
            np.mean(self.weights), np.array(w_means), np.array(w_stds)
        )
        if plot_name is None:
            plot_name = "Weights: " + ",".join([f"{x:.3f}" for x in self.weights])

        fig = go.Figure(
            data=go.Contour(
                x=w_means,
                y=w_stds,
                z=agg_values,
                zmin=0.0,
                zmax=1.0,
                colorscale=color,
                contours_coloring="heatmap",
                line_width=0,
                colorbar=dict(
                    title=str(self.agg_fn.letter),
                    titleside="right",
                    outlinewidth=1,
                    title_font_size=22,
                    tickfont_size=15,
                ),
                hoverinfo="none",
            ),
            layout=go.Layout(
                title=go.layout.Title(text=plot_name, font_size=30),
                title_x=0.5,
                xaxis_range=[0.0, np.mean(self.weights)],
                yaxis_range=[0.0, np.mean(self.weights)/2],
            ),
        )

        fig.update_xaxes(
            title_text="M: mean",
            title_font_size=22,
            tickfont_size=15,
            tickmode="auto",
            showline=True,
            linewidth=1.25,
            linecolor="black",
            minor=dict(ticklen=6, ticks="inside", tickcolor="black", showgrid=True),
        )
        fig.update_yaxes(
            title_text="SD: std",
            title_font_size=22,
            tickfont_size=15,
            showline=True,
            linewidth=1.25,
            linecolor="black",
            minor=dict(ticklen=6, ticks="inside", tickcolor="black", showgrid=True),
        )

        # calculate upper perimeter
        if len(set(self.weights)) == 1:

            def max_std(m, n):
                floor_mn = np.floor(m * n)
                nm = n * m
                value_under_sqrt = n * (floor_mn + (floor_mn - nm) ** 2) - nm**2
                return np.sqrt(value_under_sqrt) / n

            means = np.linspace(0, 1, 10000)
            half_perimeter = max_std(means[:len(means)//2], self.m)
            perimeter = np.concatenate((half_perimeter, np.flip(half_perimeter)))   
        else:
            quality_exact = {
                2: 125,
                3: 100,
                4: 75,
            }
            means = np.linspace(0, np.mean(self.weights), quality_exact.get(self.m, 50))
            half_perimeter = [self.max_std_calculator(mean, self.weights) for mean in means[:len(means)//2]]
            perimeter = np.concatenate((half_perimeter, np.flip(half_perimeter)))
            
        # draw upper perimeter
        fig.add_trace(
            go.Scatter(
                x=means,
                y=perimeter,
                mode="lines",
                showlegend=False,
                hoverinfo="none",
                line_color="black",
            )
        )

        # fill between the line and the std = 0.5
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0.5, 0.5],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255, 255, 255, 1)",
                showlegend=False,
                hoverinfo="none",
                line_color="white",
            )
        )

        # fill from the end of the graph to mean = 1
        fig.add_trace(
            go.Scatter(
                x=[max(means), max(means), 1],
                y=[0, 0.5, 0.5],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255, 255, 255, 1)",
                showlegend=False,
                hoverinfo="none",
                line_color="white",
            )
        )

        self.plot_background = go.Figure(fig)
        values = self.X_new[str(self.agg_fn.letter)].to_list()

        ### plot the ranked data
        custom = []
        for i in self.X_new.index.values:
            custom.append(1 + self._ranked_alternatives.index(i))

        fig.add_trace(
            go.Scatter(
                x=self.X_new["Mean"].tolist(),
                y=self.X_new["Std"].tolist(),
                showlegend=False,
                mode="markers",
                marker=dict(color="black", size=10),
                customdata=np.stack((custom, values), axis=1),
                text=self.X_new.index.values,
                hovertemplate="<b>ID</b>: %{text}<br>"
                + "<b>Rank</b>: %{customdata[0]:f}<br>"
                + f"<b>{str(self.agg_fn.letter)}</b>: " "%{customdata[1]:f}<br>"
                + "<extra></extra>",
            )
        )
        ### add names
        if show_names == True:
            for i, label in enumerate(self.X_new.index.values):
                fig.add_annotation(
                    x=self.X_new["Mean"].tolist()[i] + 0.01,
                    y=self.X_new["Std"].tolist()[i] + 0.01,
                    text=label,
                    showarrow=False,
                    font=dict(size=12),
                )
        return fig

    def __update_for_plot(self, id, changes, change_number):

        if "Mean" in changes.columns and "Std" in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc["NEW " + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc["NEW " + id, "Mean"] += changes["Mean"].values[0]
            self.X_newPoint.loc["NEW " + id, "Std"] += changes["Std"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(
                np.mean(self.weights),
                self.X_newPoint.loc["NEW " + id, "Mean"],
                self.X_newPoint.loc["NEW " + id, "Std"],
            )
            self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)] = agg_value
        elif "Mean" in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc["NEW " + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc["NEW " + id, "Mean"] += changes["Mean"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(
                np.mean(self.weights),
                self.X_newPoint.loc["NEW " + id, "Mean"],
                self.X_newPoint.loc["NEW " + id, "Std"],
            )
            self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)] = agg_value
        elif "Std" in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc["NEW " + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc["NEW " + id, "Std"] += changes["Std"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(
                np.mean(self.weights),
                self.X_newPoint.loc["NEW " + id, "Mean"],
                self.X_newPoint.loc["NEW " + id, "Std"],
            )
            self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)] = agg_value
        else:
            row_to_add = changes.iloc[change_number]
            result = self.X.loc[id].add(row_to_add, fill_value=0)
            new_row = pd.Series(result, name="NEW " + id)
            self.X_newPoint = self.X.copy()
            self.X_newPoint = self.X_newPoint.append(new_row)
            self.X_newPoint = self.__normalize_data(self.X_newPoint)
            w_means, w_stds = self.transform_US_to_wmsd(np.array(self.X_newPoint))
            agg_values = self.agg_fn.TOPSIS_calculation(
                np.mean(self.weights), w_means, w_stds
            )
            self.X_newPoint["Mean"] = w_means
            self.X_newPoint["Std"] = w_stds
            self.X_newPoint[str(self.agg_fn.letter)] = agg_values
        return self.X_newPoint

    def plot_improvement(self, id, changes, show_names=False, change_number=0):
        """Plots positions of alternatives in WMSD space and visualize the change after applying improvement action. 
        Parameters
        ----------
        id : string
            String value containing the name of the alternative to improve.
        change : pandas Data-frame
            Data-frame with the improvement action changes
        show_names : bool, optional
            Boolean value, if true, then points labels are showed on a plot.
                (default: False)
        change_number : int, optional
            Integer value indicating which change should be applied to plot, if there are more than 1.
            (default: 0)
        Returns
        -------
        Plot as an plotly figure.
        """
        self.__update_for_plot(id, changes, change_number)
        old_rank = (
            self.X_new.sort_values(by=str(self.agg_fn.letter), ascending=False).index.get_loc(id) + 1
        )
        old_value = self.X_new.loc[id, str(self.agg_fn.letter)]
        fig = go.Figure(self.plot_background)

        ### add old point
        fig.add_trace(
            go.Scatter(
                x=[self.X_new.loc[id, "Mean"]],
                y=[self.X_new.loc[id, "Std"]],
                showlegend=False,
                mode="markers",
                marker=dict(color="black", size=10),
                customdata=np.stack(([old_rank], [old_value]), axis=1),
                text=["OLD " + id],
                hovertemplate="<b>ID</b>: %{text}<br>"
                + "<b>Old Rank</b>: %{customdata[0]:f}<br>"
                + "<b>New Rank</b>: -<br>"
                + f"<b>{str(self.agg_fn)[16]}</b>: " "%{customdata[1]:f}<br>"
                + "<extra></extra>",
            )
        )
        ### add name for old point
        if show_names == True:
            fig.add_annotation(
                x=self.X_new.loc[id, "Mean"] + 0.01,
                y=self.X_new.loc[id, "Std"] + 0.01,
                text="OLD " + id,
                showarrow=False,
                font=dict(size=12),
            )

        self.X_newPoint = self.X_newPoint.drop(index=(id))
        self.X_newPoint = self.X_newPoint.sort_values(by=str(self.agg_fn.letter), ascending=False)
        new_rank = self.X_newPoint.index.get_loc("NEW " + id) + 1
        new_value = self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)]

        numbers = [i for i in range(1, self.n + 1)]
        custom0 = numbers.copy()
        custom0.remove(old_rank)
        custom1 = numbers.copy()
        custom1.remove(new_rank)

        ### add new point
        fig.add_trace(
            go.Scatter(
                x=[self.X_newPoint.loc["NEW " + id, "Mean"]],
                y=[self.X_newPoint.loc["NEW " + id, "Std"]],
                showlegend=False,
                mode="markers",
                marker=dict(color="white", size=10, line=dict(color='black', width=2)),
                customdata=np.stack(([new_rank], [new_value]), axis=1),
                text=["NEW " + id],
                hovertemplate="<b>ID</b>: %{text}<br>"
                + "<b>Old Rank</b>: -<br>"
                + "<b>New Rank</b>: %{customdata[0]:f}<br>"
                + f"<b>{str(self.agg_fn)[16]}</b>: " "%{customdata[1]:f}<br>"
                + "<extra></extra>",
            )
        )
        ### add names for new point and other points
        if show_names == True:
            for i, label in enumerate(self.X_newPoint.index.values):
                fig.add_annotation(
                    x=self.X_newPoint["Mean"].tolist()[i] + 0.01,
                    y=self.X_newPoint["Std"].tolist()[i] + 0.01,
                    text=label,
                    showarrow=False,
                    font=dict(size=12),
                )

        ### add arrow between old point and new point
        fig.add_annotation(
            x=self.X_newPoint.loc["NEW " + id, "Mean"],
            y=self.X_newPoint.loc["NEW " + id, "Std"],
            ax=self.X_new.loc[id, "Mean"],
            ay=self.X_new.loc[id, "Std"],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor='black'
        )

        self.X_newPoint = self.X_newPoint.drop(index=("NEW " + id))
        values = self.X_newPoint[str(self.agg_fn.letter)].to_list()

        ### add other points
        fig.add_trace(
            go.Scatter(
                x=self.X_newPoint["Mean"].tolist(),
                y=self.X_newPoint["Std"].tolist(),
                showlegend=False,
                mode="markers",
                marker=dict(color="black", size=10),
                customdata=np.stack((custom0, custom1, values), axis=1),
                text=self.X_newPoint.index.values,
                hovertemplate="<b>ID</b>: %{text}<br>"
                + "<b>Old Rank</b>: %{customdata[0]:f}<br>"
                + "<b>New Rank</b>: %{customdata[1]:f}<br>"
                + f"<b>{str(self.agg_fn)[16]}</b>: " "%{customdata[2]:f}<br>"
                + "<extra></extra>",
            )
        )
        return fig

    def show_ranking(self, mode="standard", first=1, last=None):
        """Displays the TOPSIS ranking
        Parameters
        ----------
        
        mode : 'minimal'/'standard'/'full', optional
            Way of display of the ranking. If mode='minimal', then only positions
            of ranked alternatives will be displayed. If mode='standard' then additionally
            all criteria values will be showed. If mode='full', then apart of criteria
            values also values of mean, standard deviation and aggregation function will be displayed.
            (default 'standard')
        first : int, optional
            Rank from which the ranking should be displayed.
            (default 1)
        first : int, optional
            Rank to which the ranking should be displayed.
            (default None)
        """
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
            ranking.loc[alternative, "Rank"] = (
                self._ranked_alternatives.index(alternative) + 1
            )

        ranking = ranking.sort_values(by=["Rank"])
        # ranking = ranking.loc[max(first-1, 0):last]
        ranking = ranking[(first - 1) : last]

        if isinstance(mode, str):
            if mode == "minimal":
                display(ranking["Rank"])
            elif mode == "standard":
                display(ranking.drop(["Mean", "Std", str(self.agg_fn.letter)], axis=1))
            elif mode == "full":
                display(ranking)
            else:
                raise ValueError(
                    "Invalid value at 'mode': must be a string (minimal, standard, or full)."
                )
            return

        display(ranking.drop(["Mean", "Std", str(self.agg_fn.letter)], axis=1))
        return

    def improvement(
        self,
        function_name,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon=0.000001,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if type(alternative_to_improve) == int:
            alternative_to_improve = self.X_new.loc[
                self._ranked_alternatives[alternative_to_improve]
            ].copy()
        elif type(alternative_to_improve) == str:
            alternative_to_improve = self.X_new.loc[alternative_to_improve].copy()

        if type(alternative_to_overcome) == int:
            alternative_to_overcome = self.X_new.loc[
                self._ranked_alternatives[alternative_to_overcome]
            ].copy()
        elif type(alternative_to_overcome) == str:
            alternative_to_overcome = self.X_new.loc[alternative_to_overcome].copy()

        func = getattr(self.agg_fn, function_name)
        return func(
            alternative_to_improve, alternative_to_overcome, epsilon, **kwargs
        )

    def __check_max_std_calculator(self, max_std_calculator):
        if isinstance(max_std_calculator, str):
            if max_std_calculator == "scip":
                from utils.max_std_calculator_scip import max_std_scip

                return max_std_scip
            elif max_std_calculator == "gurobi":
                from utils.max_std_calculator_gurobi import max_std_gurobi

                return max_std_gurobi
            else:
                raise ValueError(
                    "Invalid value at 'agg_fn': must be string (gurobi or scip) or function."
                )
        elif callable(max_std_calculator):
            return max_std_calculator
        else:
            raise ValueError(
                "Invalid value at 'agg_fn': must be string (scip or gurobi) or function."
            )

    def __check_agg_fn(self, agg_fn):
        if isinstance(agg_fn, str):
            if agg_fn == "A":
                return ATOPSIS(self)
            elif agg_fn == "I":
                return ITOPSIS(self)
            elif agg_fn == "R":
                return RTOPSIS(self)
            else:
                raise ValueError(
                    "Invalid value at 'agg_fn': must be string (A, I, or R) or class implementing TOPSISAggregationFunction."
                )
        elif issubclass(agg_fn, TOPSISAggregationFunction):
            return agg_fn(self)
        else:
            raise ValueError(
                "Invalid value at 'agg_fn': must be string (A, I, or R) or class implementing TOPSISAggregationFunction."
            )

    def __check_weights(self, weights):
        if isinstance(weights, list):
            return weights

        elif isinstance(weights, dict):
            return self.__dict_to_list(weights)

        elif weights is None:
            return np.ones(self.m)

        else:
            raise ValueError(
                "Invalid value at 'weights': must be a list or a dictionary"
            )

    def __check_objectives(self, objectives):
        if isinstance(objectives, list):
            return objectives
        elif isinstance(objectives, str):
            return np.repeat(objectives, self.m)
        elif isinstance(objectives, dict):
            return self.__dict_to_list(objectives)
        elif objectives is None:
            return np.repeat("max", self.m)
        else:
            raise ValueError(
                "Invalid value at 'objectives': must be a list or a string (gain, g, cost, c, min or max) or a dictionary"
            )

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
                raise ValueError(
                    "Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary"
                )

        elif expert_range is None:
            lower_bounds = self.X.min()
            upper_bounds = self.X.max()
            expert_range = [lower_bounds, upper_bounds]
            numpy_expert_range = np.array(expert_range).T
            return numpy_expert_range.tolist()

        else:
            raise ValueError(
                "Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary"
            )

    def __check_input(self):
        if self.X.isnull().values.any():
            raise ValueError(
                "Dataframe must not contain any none/nan values, but found at least one"
            )

        if len(self.weights) != self.m:
            raise ValueError("Invalid value 'weights'.")

        if not all(type(item) in [int, float, np.float64] for item in self.weights):
            raise ValueError(
                "Invalid value 'weights'. Expected numerical value (int or float)."
            )

        if not all(item >= 0 for item in self.weights):
            raise ValueError(
                "Invalid value 'weights'. Expected value must be non-negative."
            )

        if not any(item > 0 for item in self.weights):
            raise ValueError(
                "Invalid value 'weights'. At least one weight must be positive."
            )

        if len(self.objectives) != self.m:
            raise ValueError("Invalid value 'objectives'.")

        if not all(item in ["min", "max"] for item in self.objectives):
            raise ValueError(
                "Invalid value at 'objectives'. Use 'min', 'max', 'gain', 'cost', 'g' or 'c'."
            )

        if len(self.expert_range) != len(self.objectives):
            raise ValueError(
                "Invalid value at 'expert_range'. Length of should be equal to number of criteria."
            )

        for col in self.expert_range:
            if len(col) != 2:
                raise ValueError(
                    "Invalid value at 'expert_range'. Every criterion has to have minimal and maximal value."
                )
            if not all(type(item) in [int, float] for item in col):
                raise ValueError(
                    "Invalid value at 'expert_range'. Expected numerical value (int or float)."
                )
            if col[0] > col[1]:
                raise ValueError(
                    "Invalid value at 'expert_range'. Minimal value  is bigger then maximal value."
                )

        lower_bound = np.array(self.X.min()).tolist()
        upper_bound = np.array(self.X.max()).tolist()

        for val, mini, maxi in zip(self.expert_range, lower_bound, upper_bound):
            if not (val[0] <= mini and val[1] >= maxi):
                raise ValueError(
                    "Invalid value at 'expert_range'. All values from original data must be in a range of expert_range."
                )

    def __check_input_after_transform(self, X):
        n = X.shape[0]
        m = X.shape[1]

        if X.isnull().values.any():
            raise ValueError(
                "Dataframe must not contain any none/nan values, but found at least one"
            )

        if self.m != m:
            raise ValueError(
                "Invalid number of columns. Number of criteria must be the same as in previous dataframe."
            )

        if not all(X.columns.values == self.X.columns.values):
            raise ValueError(
                "New dataset must have the same columns as the dataset used to fit WMSDTransformer"
            )

        lower_bound = np.array(X.min()).tolist()
        upper_bound = np.array(X.max()).tolist()

        for val, mini, maxi in zip(self.expert_range, lower_bound, upper_bound):
            if not (val[0] <= mini and val[1] >= maxi):
                raise ValueError(
                    "Invalid value at 'expert_range'. All values from original data must be in a range of expert_range."
                )

    def __check_show_ranking(self, first, last):
        if isinstance(first, int):
            if first < 1 or first > len(self.X_new.index):
                raise ValueError(
                    f"Invalid value at 'first': must be in range [1:{len(self.X_new.index)}]"
                )
        else:
            raise TypeError("Invalid type of 'first': must be an int")

        if isinstance(last, int):
            if last < 1 or last > len(self.X_new.index):
                raise ValueError(
                    f"Invalid value at 'last': must be in range [1:{len(self.X_new.index)}]"
                )
        else:
            raise TypeError("Invalid type of 'last': must be an int")

        if last < first:
            raise ValueError("'first' must be not greater than 'last'")

    def __normalize_data(self, data):
        c = 0
        for col in data.columns:
            data[col] = (data[col] - self.expert_range[c][0]) / (
                self.expert_range[c][1] - self.expert_range[c][0]
            )
            c += 1

        for i in range(self.m):
            if self.objectives[i] == "min":
                data[data.columns[i]] = 1 - data[data.columns[i]]

        return data

    def __normalize_weights(self, weights):
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

        self.X_new["Mean"] = wm
        self.X_new["Std"] = wsd

    def __ranking(self):
        data__ = self.X_new.copy()
        data__ = data__.sort_values(by=str(self.agg_fn.letter), ascending=False)
        arranged = data__.index.tolist()
        return arranged

    def __dict_to_list(self, dictionary):
        new_list = []

        for col_name in self.X.columns:
            new_list.append(dictionary[col_name])

        return new_list


class TOPSISAggregationFunction(ABC):
    """
    Class description
    ...
    Attributes
    ----------
    attribute : type
        description
    """

    def __init__(self, wmsd_transformer):
        self.wmsd_transformer = wmsd_transformer

    @abstractmethod
    def TOPSIS_calculation(self, w, wm, wsd):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        pass

    @abstractmethod
    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        pass

    def improvement_mean(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        allow_std=False,
        solutions_number = None,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        m_start = alternative_to_improve["Mean"]
        m_boundary = w
        std_start = alternative_to_improve["Std"]
        if (
            self.TOPSIS_calculation(w, m_boundary, alternative_to_improve["Std"])
            < alternative_to_overcome[str(self.letter)]
        ):
            return None
        else:
            change = (m_boundary - alternative_to_improve["Mean"]) / 2
            actual_aggfn = self.TOPSIS_calculation(
                w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
            )
            while True:
                if actual_aggfn >= alternative_to_overcome[str(self.letter)]:
                    if (
                        actual_aggfn - alternative_to_overcome[str(self.letter)]
                        > epsilon
                    ):
                        alternative_to_improve["Mean"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                    else:
                        break
                else:
                    alternative_to_improve["Mean"] += change
                    actual_aggfn = self.TOPSIS_calculation(
                        w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                    )
                    if actual_aggfn >= alternative_to_overcome[str(self.letter)]:
                        change = change / 2
            if alternative_to_improve["Std"] <= self.wmsd_transformer.max_std_calculator(
                alternative_to_improve["Mean"], self.wmsd_transformer.weights
            ):
                if solutions_number is None:
                    return pd.DataFrame(
                        [alternative_to_improve["Mean"] - m_start], columns=["Mean"]
                    )
                else:
                    inverse_solutions = self.wmsd_transformer.inverse_transform(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                    reduced_solutions = self.reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                    result = pd.DataFrame(reduced_solutions, columns=alternative_to_improve.index[:-3])
            elif allow_std:
                alternative_to_improve["Std"] = self.wmsd_transformer.max_std_calculator(
                    alternative_to_improve["Mean"], self.wmsd_transformer.weights
                )
                actual_aggfn = self.TOPSIS_calculation(
                    w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                )
                if actual_aggfn >= alternative_to_overcome[str(self.letter)]:
                    if solutions_number is None:
                        return pd.DataFrame(
                            [
                                [
                                    alternative_to_improve["Mean"] - m_start,
                                    alternative_to_improve["Std"] - std_start,
                                ]
                            ],
                            columns=["Mean", "Std"],
                        )
                    else:
                        inverse_solutions = self.wmsd_transformer.inverse_transform(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                        reduced_solutions = self.reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                        result = pd.DataFrame(reduced_solutions, columns=alternative_to_improve.index[:-3])
                else:
                    if solutions_number is None:
                        return pd.DataFrame(
                            [
                                [
                                    alternative_to_improve["Mean"] - m_start,
                                    alternative_to_improve["Std"] - std_start,
                                ]
                            ],
                            columns=["Mean", "Std"],
                        ) + self.improvement_mean(
                            alternative_to_improve,
                            alternative_to_overcome,
                            epsilon,
                            allow_std,
                            **kwargs,
                        )
                    else:
                        return self.improvement_mean(
                            alternative_to_improve,
                            alternative_to_overcome,
                            epsilon,
                            allow_std,
                            solutions_number,
                            **kwargs,
                        )
            else:
                while alternative_to_improve["Mean"] <= m_boundary:
                    if alternative_to_improve[
                        "Std"
                    ] <= self.wmsd_transformer.max_std_calculator(
                        alternative_to_improve["Mean"], self.wmsd_transformer.weights
                    ):
                        if solutions_number is None:
                            return pd.DataFrame(
                                [alternative_to_improve["Mean"] - m_start], columns=["Mean"]
                            )
                        else:
                            inverse_solutions = self.wmsd_transformer.inverse_transform(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                            reduced_solutions = self.reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                            result =  pd.DataFrame(reduced_solutions, columns=alternative_to_improve.index[:-3])
                            break
                    alternative_to_improve["Mean"] += epsilon
                else:
                    return None
            objectives = self.wmsd_transformer.objectives
            value_range = self.wmsd_transformer._value_range
            result -= alternative_to_improve[:-3]
            for i in result.index:
                for j in range(len(result.columns)):
                    if result[result.columns[j]][i] == 0:
                        continue
                    elif objectives[j] == "max":
                        result[result.columns[j]][i] = (
                            value_range[j] * result[result.columns[j]][i]
                        )
                    else:
                        result[j][i] = (
                            -value_range[j] * result[result.columns[j]][i]
                        )
            return result

    def __check_boundary_values(
        self, alternative_to_improve, features_to_change, boundary_values
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if boundary_values is None:
            boundary_values = np.ones(len(features_to_change))
        else:
            if len(features_to_change) != len(boundary_values):
                raise ValueError(
                    "Invalid value at 'boundary_values': must be same length as 'features_to_change'"
                )
            for i in range(len(features_to_change)):
                col = self.wmsd_transformer.X_new.columns.get_loc(features_to_change[i])
                if (
                    boundary_values[i] < self.wmsd_transformer.expert_range[col][0]
                    or boundary_values[i] > self.wmsd_transformer.expert_range[col][1]
                ):
                    raise ValueError(
                        "Invalid value at 'boundary_values': must be between defined 'expert_range'"
                    )
                else:
                    boundary_values[i] = (
                        boundary_values[i] - self.wmsd_transformer.expert_range[col][0]
                    ) / (
                        self.wmsd_transformer.expert_range[col][1]
                        - self.wmsd_transformer.expert_range[col][0]
                    )
                    if self.wmsd_transformer.objectives[col] == "min":
                        boundary_values[i] = 1 - boundary_values[i]
                    if (
                        alternative_to_improve[features_to_change[i]]
                        > boundary_values[i]
                    ):
                        raise ValueError(
                            "Invalid value at 'boundary_values': must be better or equal to improving alternative values"
                        )
        return np.array(boundary_values)

    def improvement_features(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        features_to_change,
        boundary_values=None,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'"
            )
        boundary_values = self.__check_boundary_values(
            alternative_to_improve, features_to_change, boundary_values
        )

        AggFn = alternative_to_improve[str(self.letter)]
        alternative_to_improve = alternative_to_improve.drop(
            labels=["Mean", "Std", str(self.letter)]
        )
        improvement_start = alternative_to_improve.copy()
        feature_pointer = 0
        w = self.wmsd_transformer.weights
        value_range = self.wmsd_transformer._value_range
        objectives = self.wmsd_transformer.objectives

        is_improvement_satisfactory = False

        s = np.sqrt(sum(w * w)) / np.mean(w)
        for i, k in zip(features_to_change, boundary_values):
            alternative_to_improve[i] = k
            mean, std = self.wmsd_transformer.transform_US_to_wmsd(
                [alternative_to_improve]
            )
            AggFn = self.TOPSIS_calculation(np.mean(w), mean, std)

            if AggFn < alternative_to_overcome[str(self.letter)]:
                continue

            alternative_to_improve[i] = 0.5 * k
            mean, std = self.wmsd_transformer.transform_US_to_wmsd(
                [alternative_to_improve]
            )
            AggFn = self.TOPSIS_calculation(np.mean(w), mean, std)
            change_ratio = 0.25 * k
            while True:
                if AggFn < alternative_to_overcome[str(self.letter)]:
                    alternative_to_improve[i] += change_ratio
                elif AggFn - alternative_to_overcome[str(self.letter)] > epsilon:
                    alternative_to_improve[i] -= change_ratio
                else:
                    is_improvement_satisfactory = True
                    break
                change_ratio = change_ratio / 2
                mean, std = self.wmsd_transformer.transform_US_to_wmsd(
                    [alternative_to_improve]
                )
                AggFn = self.TOPSIS_calculation(np.mean(w), mean, std)

            if is_improvement_satisfactory:
                alternative_to_improve -= improvement_start
                for j in range(len(alternative_to_improve)):
                    if alternative_to_improve[j] == 0:
                        continue
                    elif objectives[j] == "max":
                        alternative_to_improve[j] = (
                            value_range[j] * alternative_to_improve[j]
                        )
                    else:
                        alternative_to_improve[j] = (
                            -value_range[j] * alternative_to_improve[j]
                        )
                result_df = alternative_to_improve.to_frame().transpose()
                result_df = result_df.reset_index(drop=True)
                return result_df
        else:
            return None

    def improvement_genetic(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        features_to_change,
        boundary_values=None,
        allow_deterioration=False,
        popsize=None,
        n_generations=200,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        boundary_values = self.__check_boundary_values(
            alternative_to_improve, features_to_change, boundary_values
        )

        current_performances_US = (
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)])
            .to_numpy()
            .copy()
        )
        modified_criteria_subset = [
            x in features_to_change for x in self.wmsd_transformer.X.columns.tolist()
        ]

        max_possible_improved = current_performances_US.copy()
        max_possible_improved[modified_criteria_subset] = boundary_values
        w_means, w_stds = self.wmsd_transformer.transform_US_to_wmsd(
            np.array([max_possible_improved])
        )
        max_possible_agg_value = self.TOPSIS_calculation(
            np.mean(self.wmsd_transformer.weights), w_means, w_stds
        ).item()
        if max_possible_agg_value < alternative_to_overcome[str(self.letter)]:
            # print(f"Not possible to achieve target {alternative_to_overcome['AggFn']} with specified features and boundary_values. Max possible agg value is {max_possible_agg_value}")
            return None

        problem = PostFactumTopsisPymoo(
            topsis_model=self.wmsd_transformer,
            modified_criteria_subset=modified_criteria_subset,
            current_performances=current_performances_US,
            target_agg_value=alternative_to_overcome[str(self.letter)],
            upper_bounds=boundary_values,
            allow_deterioration=allow_deterioration,
        )

        if popsize is None:
            popsize_by_n_objectives = {2: 150, 3: 500, 4: 1000}
            popsize = popsize_by_n_objectives.get(len(features_to_change), 2000)

        algorithm = NSGA2(
            pop_size=popsize,
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
            save_history=False,
        )

        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", n_generations),
            seed=42,
            verbose=False,
        )

        if res.F is not None:
            improvement_actions = np.zeros(
                shape=(len(res.F), len(current_performances_US))
            )
            improvement_actions[:, modified_criteria_subset] = (
                res.F - current_performances_US[modified_criteria_subset]
            )
            improvement_actions *= np.array(self.wmsd_transformer._value_range)
            improvement_actions[
                :, np.array(self.wmsd_transformer.objectives) == "min"
            ] *= -1
            return pd.DataFrame(
                sorted(improvement_actions.tolist(), key=lambda x: x[0]),
                columns=self.wmsd_transformer.X.columns,
            )
        else:
            return None

    @staticmethod
    def __solve_quadratic_equation(a, b, c):

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None
        solution_1 = (-b + np.sqrt(discriminant)) / (2 * a)
        solution_2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return solution_1, solution_2

    @staticmethod
    def __choose_appropriate_solution(
        solution_1, solution_2, lower_bound, upper_bound, objective
    ):
        
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

    @staticmethod  
    def reduce_population_agglomerative_clustering(data_to_cluster, num_clusters):
        labels = AgglomerativeClustering(n_clusters=num_clusters, linkage="average").fit(data_to_cluster).labels_
        reduced_population = []
        for i in range(max(labels)+1):
            cluster = np.array(data_to_cluster)[labels == i]
            centroid = np.mean(cluster, axis=0)
            distances_from_centroid = np.linalg.norm(cluster-centroid, axis=1)
            closest_point = cluster[distances_from_centroid.argmin()].tolist()
            reduced_population.append(closest_point)
        return reduced_population


class PostFactumTopsisPymoo(Problem):
    """
    Class description
    ...
    Attributes
    ----------

    topsis_model : object
        Object with methods to calculate weighted means, weighted standard deviations and aggregation values (e.g. WMSDTransformer object).
    modified_criteria_subset : numpy array of bools
        description
    current_performances : object
        description
    target_agg_value : object
        description
    upper_bounds : object
        description
    allow_deterioration : object
        description
    """

    """
    topsis_model -- to jest tak naprawdę obiekt WMSDTransformer, ale to może być cokolwiek innego, byleby umiało policzyć w_means w_stds oraz agg_values (w sumie wystarczyłoby to ostatnie), ta klasa działała wcześniej bez tego całego WMSD w przestrzeni ocen, potem ją dostosowałem, bo jak implementowałem takie rzeczy do swoich badań
    modified_criteria_subset -- to jest odpowiednik z heurystyki Adama, tylko że zamiast listy stringów to jest bool-owski np.array o długości n_criteria, używam tego do szybkiego slice-owania tablic np.array
    current_performances oraz target_agg_value -- te chyba nie wymagają wyjaśnienia, zobaczcie tylko jakiego typu obiekty to są, jeśli potrzeba to wam wyjaśnię kiedyś na głosowym
    upper_bounds -- to jest odpowiednik boundary_values z heurystyki Adama, uwaga na postać i długość tej tablicy
    allow_deterioration -- to chyba można wywalić, bo tej funkcjonalności w bibliotece ostatecznie nie będzie
    """
    def __init__(
        self,
        topsis_model,
        modified_criteria_subset,
        current_performances,
        target_agg_value,
        upper_bounds,
        allow_deterioration=False,
    ):
        n_criteria = np.array(modified_criteria_subset).astype(bool).sum()
        super().__init__(
            n_var=n_criteria, n_obj=n_criteria, n_ieq_constr=1, vtype=float
        )

        self.topsis_model = topsis_model
        self.mean_of_weights = np.mean(self.topsis_model.weights)
        self.modified_criteria_subset = np.array(modified_criteria_subset).astype(bool)
        self.current_performances = current_performances.copy()
        self.target_agg_value = target_agg_value

        # Lower and upper bounds in Utility Space
        self.xl = (
            np.zeros(n_criteria)
            if allow_deterioration
            else self.current_performances[self.modified_criteria_subset]
        )
        self.xu = upper_bounds

    def _evaluate(self, x, out, *args, **kwargs):
        # In Utility Space variables and objectives are the same values
        out["F"] = x.copy()  # this copy might be redundant

        # Topsis target constraint
        modified_performances = np.repeat(
            [self.current_performances], repeats=len(x), axis=0
        )
        modified_performances[
            :, self.modified_criteria_subset
        ] = x.copy()  # this copy might be redundant
        w_means, w_stds = self.topsis_model.transform_US_to_wmsd(modified_performances)
        agg_values = self.topsis_model.agg_fn.TOPSIS_calculation(
            self.mean_of_weights, w_means, w_stds
        )
        g1 = (
            self.target_agg_value - agg_values
        )  # In Pymoo positive values indicate constraint violation
        out["G"] = np.array([g1])


class ATOPSIS(TOPSISAggregationFunction):
    """
    Class description
    ...
    Attributes
    ----------
    attribute : type
        description
    """

    def __init__(self, wmsd_transformer):
        super().__init__(wmsd_transformer)
        self.letter = 'A'

    def TOPSIS_calculation(self, w, wm, wsd):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        return np.sqrt(wm * wm + wsd * wsd) / w

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        """Exact algorithm dedicated to the aggregation `A` for achieving the target by modifying the performance on a single criterion."""
        performances_US = (
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)])
            .to_numpy()
            .copy()
        )
        performances_CS = (
            performances_US * self.wmsd_transformer._value_range
            + self.wmsd_transformer._lower_bounds
        )
        weights = self.wmsd_transformer.weights
        target_agg_value = (
            alternative_to_overcome[str(self.letter)] + epsilon / 2
        ) * np.linalg.norm(weights)

        modified_criterion_idx = list(
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)]).index
        ).index(feature_to_change)
        criterion_range = self.wmsd_transformer._value_range[modified_criterion_idx]
        lower_bound = self.wmsd_transformer._lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.wmsd_transformer.objectives[modified_criterion_idx]

        # Negative Ideal Solution (utility space)
        NIS = np.zeros_like(performances_US)

        v_ij = performances_US * weights
        j = modified_criterion_idx

        v_ij_excluding_j = np.delete(v_ij, j)
        NIS_excluding_j = np.delete(NIS, j)

        a = 1
        b = -2 * NIS[j]
        c = (
            NIS[j] ** 2
            + np.sum((v_ij_excluding_j - NIS_excluding_j) ** 2)
            - target_agg_value**2
        )

        solutions = TOPSISAggregationFunction.__solve_quadratic_equation(
            a, b, c
        )  # solutions are new performances in VS, not modifications
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = ((solutions[0] / weights[j]) * criterion_range) + lower_bound
            solution_2 = ((solutions[1] / weights[j]) * criterion_range) + lower_bound

            # solution -- new performances in CS
            solution = TOPSISAggregationFunction.__choose_appropriate_solution(
                solution_1, solution_2, lower_bound, upper_bound, objective
            )
            if solution is None:
                return None
            else:
                feature_modification = solution - performances_CS[j]
                modification_vector = np.zeros_like(performances_US)
                modification_vector[modified_criterion_idx] = feature_modification
                result_df = pd.DataFrame(
                    [modification_vector], columns=self.wmsd_transformer.X.columns
                )
                return result_df

    def improvement_std(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        solutions_number = None,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        sd_boundary = self.wmsd_transformer.max_std_calculator(
            alternative_to_improve["Mean"], self.wmsd_transformer.weights
        )
        if (
            self.TOPSIS_calculation(w, alternative_to_improve["Mean"], sd_boundary)
            < alternative_to_overcome[str(self.letter)]
        ):
            return None
        else:
            change = (sd_boundary - alternative_to_improve["Std"]) / 2
            actual_aggfn = self.TOPSIS_calculation(
                w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
            )
            while True:
                if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                    if (
                        actual_aggfn - alternative_to_overcome[str(self.letter)]
                        > epsilon
                    ):
                        alternative_to_improve["Std"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                    else:
                        break
                else:
                    alternative_to_improve["Std"] += change
                    change = change / 2
                    actual_aggfn = self.TOPSIS_calculation(
                        w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                    )
            if solutions_number is None:
                return pd.DataFrame(
                    [alternative_to_improve["Std"] - std_start], columns=["Std"]
                )
            else:
                inverse_solutions = self.wmsd_transformer.inverse_transform(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                reduced_solutions = self.reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                result = pd.DataFrame(reduced_solutions, columns=alternative_to_improve.index[:-3])
            objectives = self.wmsd_transformer.objectives
            value_range = self.wmsd_transformer._value_range
            result -= alternative_to_improve[:-3]
            for i in result.index:
                for j in range(len(result.columns)):
                    if result[result.columns[j]][i] == 0:
                        continue
                    elif objectives[j] == "max":
                        result[result.columns[j]][i] = (
                            value_range[j] * result[result.columns[j]][i]
                        )
                    else:
                        result[j][i] = (
                            -value_range[j] * result[result.columns[j]][i]
                        )
            return result


class ITOPSIS(TOPSISAggregationFunction):
    """
    Class description
    ...
    Attributes
    ----------
    attribute : type
        description
    """

    def __init__(self, wmsd_transformer):
        super().__init__(wmsd_transformer)
        self.letter = 'I'

    def TOPSIS_calculation(self, w, wm, wsd):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        return 1 - np.sqrt((w - wm) * (w - wm) + wsd * wsd) / w

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        """Exact algorithm dedicated to the aggregation `I` for achieving the target by modifying the performance on a single criterion."""
        performances_US = (
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)])
            .to_numpy()
            .copy()
        )
        performances_CS = (
            performances_US * self.wmsd_transformer._value_range
            + self.wmsd_transformer._lower_bounds
        )
        weights = self.wmsd_transformer.weights
        target_agg_value = (
            1 - (alternative_to_overcome[str(self.letter)] + epsilon / 2)
        ) * np.linalg.norm(weights)

        modified_criterion_idx = list(
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)]).index
        ).index(feature_to_change)
        criterion_range = self.wmsd_transformer._value_range[modified_criterion_idx]
        lower_bound = self.wmsd_transformer._lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.wmsd_transformer.objectives[modified_criterion_idx]

        # Positive Ideal Solution (utility space)
        PIS = weights

        v_ij = performances_US * weights
        j = modified_criterion_idx

        v_ij_excluding_j = np.delete(v_ij, j)
        PIS_excluding_j = np.delete(PIS, j)

        a = 1
        b = -2 * PIS[j]
        c = (
            PIS[j] ** 2
            + np.sum((v_ij_excluding_j - PIS_excluding_j) ** 2)
            - target_agg_value**2
        )

        solutions = TOPSISAggregationFunction.__solve_quadratic_equation(
            a, b, c
        )  # solutions are new performances in VS, not modifications
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = ((solutions[0] / weights[j]) * criterion_range) + lower_bound
            solution_2 = ((solutions[1] / weights[j]) * criterion_range) + lower_bound

            # solution -- new performances in CS
            solution = TOPSISAggregationFunction.__choose_appropriate_solution(
                solution_1, solution_2, lower_bound, upper_bound, objective
            )
            if solution is None:
                return None
            else:
                feature_modification = solution - performances_CS[j]
                modification_vector = np.zeros_like(performances_US)
                modification_vector[modified_criterion_idx] = feature_modification
                result_df = pd.DataFrame(
                    [modification_vector], columns=self.wmsd_transformer.X.columns
                )
                return result_df

    def improvement_std(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        solutions_number = None,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        sd_boundary = self.wmsd_transformer.max_std_calculator(
            alternative_to_improve["Mean"], self.wmsd_transformer.weights
        )
        if (
            self.TOPSIS_calculation(w, alternative_to_improve["Mean"], 0)
            < alternative_to_overcome[str(self.letter)]
        ):
            return None
        else:
            change = alternative_to_improve["Std"] / 2
            actual_aggfn = self.TOPSIS_calculation(
                w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
            )
            while True:
                if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                    if (
                        actual_aggfn - alternative_to_overcome[str(self.letter)]
                        > epsilon
                    ):
                        alternative_to_improve["Std"] += change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                    else:
                        break
                else:
                    alternative_to_improve["Std"] -= change
                    change = change / 2
                    actual_aggfn = self.TOPSIS_calculation(
                        w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                    )
            if solutions_number is None:
                return pd.DataFrame(
                    [alternative_to_improve["Std"] - std_start], columns=["Std"]
                )
            else:
                inverse_solutions = self.wmsd_transformer.inverse_transform(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                reduced_solutions = self.reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                result = pd.DataFrame(reduced_solutions, columns=alternative_to_improve.index[:-3])
            objectives = self.wmsd_transformer.objectives
            value_range = self.wmsd_transformer._value_range
            result -= alternative_to_improve[:-3]
            for i in result.index:
                for j in range(len(result.columns)):
                    if result[result.columns[j]][i] == 0:
                        continue
                    elif objectives[j] == "max":
                        result[result.columns[j]][i] = (
                            value_range[j] * result[result.columns[j]][i]
                        )
                    else:
                        result[j][i] = (
                            -value_range[j] * result[result.columns[j]][i]
                        )
            return result


class RTOPSIS(TOPSISAggregationFunction):
    """
    Class description
    ...
    Attributes
    ----------
    attribute : type
        description
    """

    def __init__(self, wmsd_transformer):
        super().__init__(wmsd_transformer)
        self.letter = 'R'

    def TOPSIS_calculation(self, w, wm, wsd):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        return np.sqrt(wm * wm + wsd * wsd) / (
            np.sqrt(wm * wm + wsd * wsd) + np.sqrt((w - wm) * (w - wm) + wsd * wsd)
        )

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        """Exact algorithm dedicated to the aggregation `R` for achieving the target by modifying the performance on a single criterion."""
        performances_US = (
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)])
            .to_numpy()
            .copy()
        )
        performances_CS = (
            performances_US * self.wmsd_transformer._value_range
            + self.wmsd_transformer._lower_bounds
        )
        weights = self.wmsd_transformer.weights
        target_agg_value = alternative_to_overcome[str(self.letter)] + epsilon / 2

        modified_criterion_idx = list(
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)]).index
        ).index(feature_to_change)
        criterion_range = self.wmsd_transformer._value_range[modified_criterion_idx]
        lower_bound = self.wmsd_transformer._lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.wmsd_transformer.objectives[modified_criterion_idx]

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
        p = k * np.sum((v_ij_excluding_j - PIS_excluding_j) ** 2) - np.sum(
            (v_ij_excluding_j - NIS_excluding_j) ** 2
        )

        a = (1 - k) * (weights[j] / criterion_range) ** 2
        b = (
            2
            * (weights[j] / criterion_range)
            * (v_ij[j] - NIS[j] - k * (v_ij[j] - PIS[j]))
        )
        c = (v_ij[j] - NIS[j]) ** 2 - k * (v_ij[j] - PIS[j]) ** 2 - p

        solutions = TOPSISAggregationFunction.__solve_quadratic_equation(
            a, b, c
        )  # solutions are performance modifications in CS !!!
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = solutions[0] + performances_CS[j]
            solution_2 = solutions[1] + performances_CS[j]

        # solution -- new performances in CS
        solution = TOPSISAggregationFunction.__choose_appropriate_solution(
            solution_1, solution_2, lower_bound, upper_bound, objective
        )
        if solution is None:
            return None
        else:
            feature_modification = solution - performances_CS[j]
            modification_vector = np.zeros_like(performances_US)
            modification_vector[modified_criterion_idx] = feature_modification
            result_df = pd.DataFrame(
                [modification_vector], columns=self.wmsd_transformer.X.columns
            )
            return result_df

    def improvement_std(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        solutions_number = None,
        **kwargs,
    ):
        """TO DO
        Parameters
        ----------
        parameter : type
            description
        Returns
        -------
        TO DO
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        sd_boundary = self.wmsd_transformer.max_std_calculator(
            alternative_to_improve["Mean"], self.wmsd_transformer.weights
        )
        if alternative_to_improve["Mean"] < w / 2:
            if (
                self.TOPSIS_calculation(w, alternative_to_improve["Mean"], sd_boundary)
                < alternative_to_overcome[str(self.letter)]
            ):
                return None
            else:
                change = (sd_boundary - alternative_to_improve["Std"]) / 2
                actual_aggfn = self.TOPSIS_calculation(
                    w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                )
                while True:
                    if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                        if (
                            actual_aggfn - alternative_to_overcome[str(self.letter)]
                            > epsilon
                        ):
                            alternative_to_improve["Std"] -= change
                            change = change / 2
                            actual_aggfn = self.TOPSIS_calculation(
                                w,
                                alternative_to_improve["Mean"],
                                alternative_to_improve["Std"],
                            )
                        else:
                            break
                    else:
                        alternative_to_improve["Std"] += change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                if solutions_number is None:
                    return pd.DataFrame(
                        [alternative_to_improve["Std"] - std_start],
                        columns=["Improvement rate"],
                        index=["Std"],
                    )
                else:
                    inverse_solutions = self.wmsd_transformer.inverse_transform(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                    reduced_solutions = self.reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                    result = pd.DataFrame(reduced_solutions, columns=alternative_to_improve.index[:-3])
        else:
            if (
                self.TOPSIS_calculation(w, alternative_to_improve["Mean"], 0)
                < alternative_to_overcome[str(self.letter)]
            ):
                return None
            else:
                change = alternative_to_improve["Std"] / 2
                actual_aggfn = self.TOPSIS_calculation(
                    w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                )
                while True:
                    if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                        if (
                            actual_aggfn - alternative_to_overcome[str(self.letter)]
                            > epsilon
                        ):
                            alternative_to_improve["Std"] += change
                            change = change / 2
                            actual_aggfn = self.TOPSIS_calculation(
                                w,
                                alternative_to_improve["Mean"],
                                alternative_to_improve["Std"],
                            )
                        else:
                            break
                    else:
                        alternative_to_improve["Std"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                if solutions_number is None:
                    return pd.DataFrame(
                        [alternative_to_improve["Std"] - std_start], columns=["Std"]
                    )
                else:
                    inverse_solutions = self.wmsd_transformer.inverse_transform(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                    reduced_solutions = self.reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                    result = pd.DataFrame(reduced_solutions, columns=alternative_to_improve.index[:-3])
            objectives = self.wmsd_transformer.objectives
            value_range = self.wmsd_transformer._value_range
            result -= alternative_to_improve[:-3]
            for i in result.index:
                for j in range(len(result.columns)):
                    if result[result.columns[j]][i] == 0:
                        continue
                    elif objectives[j] == "max":
                        result[result.columns[j]][i] = (
                            value_range[j] * result[result.columns[j]][i]
                        )
                    else:
                        result[j][i] = (
                            -value_range[j] * result[result.columns[j]][i]
                        )
            return result
