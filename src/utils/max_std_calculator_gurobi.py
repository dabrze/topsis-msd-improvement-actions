import numpy as np
import gurobipy as gp
from gurobipy import GRB


def max_std_gurobi(mean, weights):
    n_criteria = len(weights)
    w = np.array(weights)
    s = np.sqrt(sum(w * w)) / np.mean(w)

    model = gp.Model("quadratic")
    model.Params.LogToConsole = 0
    model.params.NonConvex = 2
    v = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=weights[idx]) for idx in range(n_criteria)]

    factor = gp.quicksum([v[idx] * w[idx] for idx in range(n_criteria)]) / np.sum(w ** 2)  # sum(v * w) / sum(w ** 2)
    vw = [w[idx] * factor for idx in range(n_criteria)]
    v_minus_vw = [v[idx] - vw[idx] for idx in range(n_criteria)]
    objective_function = gp.quicksum([v_minus_vw[idx] ** 2 for idx in range(n_criteria)])

    model.addConstr(gp.quicksum([vw[idx] ** 2 for idx in range(n_criteria)]) == (mean * s) ** 2)
    model.setObjective(objective_function, GRB.MAXIMIZE)
    model.optimize()

    value = max(0, model.ObjVal)
    return np.sqrt(value) / s
