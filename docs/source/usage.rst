Usage
=====
Usage examples
==============

Quickstart
----------

Creating WMSDTransformer object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create a WMSDTransformer object, you need to provide an aggregation function's name.
You can choose between **R**, standing from *Relative*, **I** (*Ideal*) and **A** (*Anti-ideal*).
.. code-block:: python

    from WMSDTransformer import WMSDTransformer
    wmsd_transformer = WMSDTransformer(agg_fn = "R")

Fitting and transforming data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is how you fit and transform data provided in form of pandas dataframe,
using the *fit_transform()* method.
.. code-block:: python

    wmsd_transformer.fit_transform(your_dataframe)

Showing TOPSIS ranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After fitting and transforming data, you can run *show_ranking()* method
to show a TOPSIS ranking.
.. code-block:: python

    wmsd_transformer.show_ranking()

Showing TOPSIS results in WMSD space
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To print a plot visualizing TOPSIS results in WMSD Space, you need to run *plot()* method.
.. code-block:: python

    wmsd_transformer.plot()

Notebooks
---------
.. _notebooks:
* `students example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/students_example.ipynb>`_
* `bus example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/bus_example.ipynb>`_
* `economic index example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/wmsd_case_studies.ipynb>`_
  
.. autosummary::
.. toctree:: generated
  
  WMSDtransformer
  

