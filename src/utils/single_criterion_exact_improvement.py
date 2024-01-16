import numpy as np


def solve_quadratic_equation(a, b, c):
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    solution_1 = (-b + np.sqrt(discriminant)) / (2 * a)
    solution_2 = (-b - np.sqrt(discriminant)) / (2 * a)
    return solution_1, solution_2


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
