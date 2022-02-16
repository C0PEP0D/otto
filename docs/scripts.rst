Scripts
=======

general usage
-------------

There are three main scripts:

- ``evaluate.py`` in the ``evaluate`` directory: for **evaluating the performance** of a policy
- ``learn.py`` in the ``learn`` directory: for **learning a neural network policy** that solves the task
- ``visualize.py`` in the ``visualize`` directory: for **visualizing a search** episode

To run any of these scripts, use::

    python script.py -i custom_params


where ``script.py`` is the script file,
and ``custom_params.py`` is a user-defined parameter file (possibly empty)
located in the local ``parameters`` directory.

Examples of parameter files called ``example*.py`` are provided in the local ``parameters`` directory.
They show *essential* parameters that can be customized for that script.
You can access the list of *all* parameters and their default values by examining
the contents of ``__defaults.py``.

Outputs are saved in the local ``outputs`` directory.


visualize.py
------------

.. automodule:: otto.visualize.visualize

evaluate.py
-----------

.. automodule:: otto.evaluate.evaluate

learn.py
--------

.. automodule:: otto.learn.learn

test.py
-------

.. automodule:: otto.test.test