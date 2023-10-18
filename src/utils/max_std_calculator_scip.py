import numpy as np
import pyscipopt as scip


def max_std_scip(mean, weights):
    n_criteria = len(weights)
    w = np.array(weights)
    s = np.sqrt(sum(w * w)) / np.mean(w)

    model = scip.Model("quadratic")
    model.hideOutput()
    v = [model.addVar(vtype="C", lb=0, ub=weights[idx]) for idx in range(n_criteria)]

    factor = scip.quicksum([v[idx] * w[idx] for idx in range(n_criteria)]) / np.sum(w ** 2)  # sum(v * w) / sum(w ** 2)
    vw = [w[idx] * factor for idx in range(n_criteria)]
    v_minus_vw = [v[idx] - vw[idx] for idx in range(n_criteria)]

    objective_variable = model.addVar(vtype="C", lb=0)
    model.setObjective(objective_variable, sense="maximize")

    # SCIP does not support nonlinear objective functions, so we need to reformulate the problem by moving the objective into a constraint.
    objective_variable_constraint = scip.quicksum([v_minus_vw[idx] ** 2 for idx in range(n_criteria)])
    model.addCons(objective_variable <= objective_variable_constraint)

    model.addCons(scip.quicksum([vw[idx] ** 2 for idx in range(n_criteria)]) == (mean * s) ** 2)
    model.optimize()

    value = max(0, model.getObjVal())
    return np.sqrt(value) / s
