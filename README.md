#TOPSIS MSD improvement actions
![](https://github.com/dabrze/topsis-msd-improvement-actions/actions/workflows/build.yml/badge.svg)

###MSDTransformer
    A class to calculate and show TOPSIS ranking of provided dataset

    Attributes
    ----------
    data : dataframe
        dataframe with data we are working on with few columns added during computations
    x : int
        number of columns
    y : int
        number of rows
    weights : np.array
        array of weights for criterias
    objectives : np.array
        array of types of criterias (gain or cost)
    expert_range
        range of values for criterias given by expert
    isFitted : bool
        a flag to tell if data is fittet

    Methods
    -------
    fit()
        fits the data to make it easier to work on it
    transform()
        performes any nesesary operation to prepare the ranking
    inverse_transform()
        returns the data to the first form
    normalizeData(data)
        normalize given data using either given expert range or min/max
    normalizeWeights(weights)
        normalize weights to make all of them at most 1
    calulateMean()
        calculates and ads mean column to dataframe
    calulateSD()
        calculates and ads standard dewiation column to dataframe
    topsis()
        calculates and ads topsis value column to dataframe
    ranking()
        creates a ranking from the data based on topsis value column
