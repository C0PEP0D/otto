Scripts
=======

There are three main scripts:

- ``evaluate.py`` in the ``evaluate`` directory: for **evaluating the performance** of a policy
- ``learn.py`` in the ``learn`` directory: for **learning a neural network policy** that solves the task
- ``visualize.py`` in the ``visualize`` directory: for **visualizing a search** episode

To run any of these scripts, use::

    python script.py -i custom


where ``script.py`` is the script file,
and ``custom.py`` is a user-defined parameter file (possibly empty) located in the local ``parameters`` directory.

Example files, ``example*.py``, shows *interesting* parameters that can be customized.
You can access the list of *all* parameters and their default values by examining
the contents of ``__defaults.py``.

Outputs are saved in the local ``outputs`` directory.


visualize
---------

.. automodule:: otto.visualize.visualize
   :members:

evaluate
---------

.. automodule:: otto.evaluate.evaluate
   :members:

learn
-----

.. automodule:: otto.learn.learn
   :members:


