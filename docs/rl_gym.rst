Gym wrapper
===========

A gym wrapper for the source-tracking POMDP is provided as part of OTTO.
`OpenAI Gym <https://gym.openai.com>`_ is the *de facto* standard for environment simulators, and is compatible with
general-purpose reinforcement learning libraries such as `Stable Baselines 3 <https://github.com/DLR-RM/stable-baselines3>`_,
`OpenAI Baselines <https://github.com/openai/baselines>`_, `RLlib <https://github.com/ray-project/ray>`_,
`CleanRL <https://github.com/vwxyzjn/cleanrl>`_, `ChainerRL <https://github.com/chainer/chainerrl>`_, and
`PFRL <https://github.com/pfnet/pfrl>`_.

We provide below a script that illustrates how to use the gym wrapper. The script can be found in ``docs/scripts``.

.. literalinclude:: scripts/demo_gym.py

