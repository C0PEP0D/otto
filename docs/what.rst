What is OTTO?
=============

OTTO (short for Odor-based Target Tracking Optimization) is a Python package to
**visualize, evaluate and learn strategies** for odor-based searches.

OTTO implements:

  - a **simulator** of the source-tracking POMDP for any number of space dimensions,
  - various **heuristic policies** including **infotaxis**,
  - a custom **deep reinforcement learning** algorithm able to yield **near-optimal policies**,
  - an **efficient algorithm to evaluate policies** using a rigorous protocol,
  - a **rendering** of searches in 1D, 2D and 3D.

OTTO has been used in a publication [Loisy2022]_.

.. figure:: gifs/3D_search.gif
  :width: 60 %
  :align: center

  Example of a 3D search with the popular infotaxis strategy.


