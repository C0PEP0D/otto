Scripts
=======

There are three main scripts (located in associated main directories)

- ``evaluate.py`` in the ``evaluate`` directory: for **evaluating the performance** of a policy
- ``learn.py`` in the ``learn`` directory: for **learning a neural network policy** that solves the task
- ``visualize.py`` in the ``visualize`` directory: for **visualizing a search** episode

Basic usage for ``script.py`` is from the terminal::

    python script.py -i custom


where ``custom.py`` is a user-defined parameter file located in the local ``parameters`` directory.

An example file, ``example.py``, shows the list of *interesting* parameters that can be customized.
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


