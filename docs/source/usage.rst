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
Below you will find notebooks which are prepared to show how to use the WMSDTransformer library
on different types of data sets.

.. _notebooks:
* `students example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/students_example.ipynb>`_
* `bus example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/bus_example.ipynb>`_
* `economic index example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/wmsd_case_studies.ipynb>`_

Students example
^^^^^^^^^^^^^^^^
Data set showed in `students example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/students_example.ipynb>`_ notebook
contains only 3 criteria, each of them is gain type and ech of them has weight equal 1.

Bus example
^^^^^^^^^^^
Data set showed in  `bus example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/bus_example.ipynb>`_ notebook
contains 4 gain type criteria and 4 cost type criteria. Each of them has weight equal 1.

Economic index example
^^^^^^^^^^^^^^^^^^^^^^
Data set showed in `economic index example <https://github.com/dabrze/topsis-msd-improvement-actions/blob/main/notebooks/wmsd_case_studies.ipynb>`_ notebook
contains 4 gain type criteria with different weights.
  

