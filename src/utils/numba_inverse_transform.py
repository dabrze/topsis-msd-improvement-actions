import time
import numpy as np
import math
import numba


@numba.jit(nopython=True)
def transform_US_to_wmsd_numba(X_US, weights):
    # transform data from Utility Space to WMSD Space
    w = weights
    squared_w = w ** 2
    sum_of_squared_weights = np.sum(squared_w)
    norm_w = np.sqrt(np.sum(squared_w))
    mean_weight = np.mean(w)

    s = norm_w / mean_weight
    v = X_US * w

    vw = np.sum(v * w) / sum_of_squared_weights * w
    w_mean = np.sqrt(np.sum(vw ** 2)) / s
    w_std = np.sqrt(np.sum((v - vw) ** 2)) / s
    return w_mean, w_std


@numba.jit(nopython=True)
def inverse_transform(target_mean, target_std, weights, std_type='==', sampling_density=None, epsilon=0.01, verbose=False):
    n_criteria = len(weights)
    if sampling_density is None:
        sampling_density = math.ceil(5000000 ** (1 / n_criteria))
    sampling_density = int(sampling_density)

    dims = [np.linspace(0, 1, sampling_density).astype(np.float32) for i in range(n_criteria)]  # the numba version of np.linspace accepts no dtype argument

    divs = []
    mods = []
    factor = 1
    for i in range((n_criteria - 1), -1, -1):
        items = len(dims[i])
        divs.insert(0, factor)
        mods.insert(0, items)
        factor *= items

    n_samples = 1
    for dim in dims:
        n_samples *= len(dim)
    if verbose:
        print(f"inverse_transform_numba: sampling_density: {sampling_density}")
        print(f"inverse_transform_numba: {n_samples} samples generated in total")

    filtered_points = []
    for i in range(0, n_samples):
        point = []
        for j in range(0, n_criteria):
            point.append(dims[j][i // divs[j] % mods[j]])
        point = np.array(point)
        wm, wsd = transform_US_to_wmsd_numba(point, weights)

        if std_type == "==":
            if abs(wm - target_mean) < epsilon and abs(wsd - target_std) < epsilon:
                filtered_points.append(point)
        elif std_type == "<=":
            if abs(wm - target_mean) < epsilon and wsd <= target_std:
                filtered_points.append(point)
        else:  # std_type == ">="
            if abs(wm - target_mean) < epsilon and wsd >= target_std:
                filtered_points.append(point)

    print(f"znaleziono {len(filtered_points)} punktów")
    if verbose:
        print(f"inverse_transform_numba: Returning {len(filtered_points)} solutions")

    return filtered_points
