What is OTTO?
=============

OTTO (short for Odor-based Target Tracking Optimization) is a Python package to
**visualize, evaluate and learn strategies** for odor-based searches.

It is aimed at researchers in biophysics, applied mathematics and robotics working on optimal strategies for olfactory searches in turbulent conditions.

OTTO implements:

  - a **simulator** of the source-tracking POMDP for any number of space dimensions,
  - various **heuristic policies** including **infotaxis**,
  - a custom **deep reinforcement learning** algorithm able to yield **near-optimal policies**,
  - a **gym wrapper** allowing the use of general-purpose reinforcement learning libraries,
  - an **efficient algorithm to evaluate policies** using a rigorous protocol,
  - a **rendering** of searches in 1D, 2D and 3D.

To facilitate the evaluation of new policies compared to existing baselines, the performance of several policies (including infotaxis and near-optimal) is summarized in a `dataset <https://doi.org/10.5281/zenodo.6125391>`_.

OTTO has been used in a publication [Loisy2022]_.

.. figure:: gifs/3D_search.gif
  :width: 60 %
  :align: center

  Example of a 3D search with the popular infotaxis strategy.


