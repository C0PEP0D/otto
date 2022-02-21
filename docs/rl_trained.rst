.. _sec-trained:

=======================
Trained neural networks
=======================

Examples of trained neural networks yielding a near-optimal policy are provided in the ``zoo`` directory.

They can be used simply by passing the name of the model to ``visualize.py`` or ``evaluate.py``.

For example, to visualize an episode using the neural network policy defined by the model ``zoo_model_1_2_2``, use::

    python3 visualize.py --input zoo_model_1_2_2


The list of available models and corresponding parameters is given below.
They are named `zoo_model_i_j_k` where i, j, k are integers associated to `N_DIMS`, `LAMBDA_OVER_DX`, `R_DT`.

.. csv-table:: Trained neural networks and their parameters.
   :file: zoo.csv
   :header-rows: 1