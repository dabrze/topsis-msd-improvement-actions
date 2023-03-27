# MSDTranformer - python library
![](https://github.com/dabrze/topsis-msd-improvement-actions/actions/workflows/build.yml/badge.svg)

New python library, that allows to create TOPSIS ranking of alternatives, visualise it in MSD space and perform improvement actions, looking for counterfactuals.

## Table of Contents
* [Technologies] (#technologies)
* [Methods] (#methods)
* [Launch] (#launch)
* [Sources] (#sources)

## Technologies
Project is created with:
* work in progress

## Methods

### fit()
This method must be run before transform() method. It fits and normalizes data. As parameters it takes:
* data : dataframe
* weights : np.array (optional)
* objectives : dictionary/string/list (optional)
* expert_range : np.array (optional)

### transform()
This method performes calculation of mean and standard deviation, and create TOPSIS ranking.

### inverse_transform()
Work in progress.

### plot()
This method visualize the ranking in MSD space.

### improvement_basic
This method prints information how much mean and standard deviation should change, to improve the position in ranking of given alternative. As parameters it takes:
* position : int
* improvement : int
* improvement_ratio : float

### improvement_features
This method prints information how much should be changed values of criteria, to improve the position in the ranking of given alternative. As parameters it takes:
* position : int
* improvement : int
* improvement_ratio : float
* features_to_change : list

## Launch
To use this library:
```
work in progress
```
## Sources
This project is inspired by the paper "MSD-space: Visualizing the Inner-Workings of TOPSIS Aggregations" by Robert Susmaga, Izabela Szczęch, Piotr Zielniewicz, Dariusz Brzeziński [PUT 2022].
