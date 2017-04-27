Networks
========

Network architectures in CN24 are defined using a JSON file.
The basic layout looks like the following example:

.. code-block:: javascript

  {
    "hyperparameters": { /* See section on hyperparameters */ },
    "net": {
      "task":        "detection",
      "input":       "conv1",
      "output":      "fc7",
      "nodes":       { /* See section on layer types */ },
      "error_layer": "square"
    },
    "data_input": { /* See section on data input */ }
  }

Hyperparameters
...............
This section controls the optimization process. The following
hyperparameters can (and *should*) be set:

* **batch_size_parallel**: Sets the fourth dimension of the
  network's input. This directly affects VRAM usage if you
  are using a GPU.
* **batch_size_sequential**: If you want to use a larger minibatch size
  than your memory would allow using **batch_size_parallel**,
  you can change **batch_size_sequential**. The effective minibatch size
  is the product of both.
* **epoch_iterations**: The number of iterations (gradient steps) per epoch.
  This is an
  arbitrary setting. If it is not set, an epoch will have one iteration
  per training sample.
* **optimization_method**: Choose the optimizer you want to use for your
  network. Currently, the following optimization methods are supported:
    * *adam*: The `Adam <https://arxiv.org/abs/1412.6980>`_ optimizer.
      It can be configured using the following hyperparameter keys:
        * **ad_step_size**, **ad_beta1** and **ad_beta2**: Matches
          the :math:`\alpha,\beta_1` and :math:`\beta_2` parameters from
          the Adam paper.
        * **ad_epsilon**: Mathces the :math:`\epsilon` parameter from the
          Adam paper.
        * **ad_sqrt_step_size**: If set to 1, the effective step size will
          be :math:`\alpha` divided by the square root of the number of iterations
          already processed.
    * *gd*: Standard stochastic gradient descent with momentum.
      Using the number of iterations :math:`t`, the effective learning
      rate is :math:`\eta (1 + \gamma t)^q`. SGD
      supports the following hyperparameter keys:
        * **learning_rate**: Sets the learning rate :math:`\eta` for gradient descent.
        * **learning_rate_exponent**: Sets the exponent :math:`q` for the
          effective learning rate.
        * **learning_rate_gamma**: Sets the coefficient :math:`\gamma` for the
          effective learning rate.
        * **gd_momentum**: Sets the momentum coefficient.
 * **l1**: The coefficient for :math:`L_1` regularization of weights.
 * **l2**: The coefficient for :math:`L_2` regularization of weights.


An example block might look like this:

.. code-block:: javascript

  "hyperparameters": {
    "testing_ratio": 1,
    "batch_size_parallel": 2, 
    "batch_size_sequential": 32, 
    "epoch_iterations": 100, 
    "l1": 0, 
    "l2": 0.0005, 
    "optimization_method": "adam",
    "ad_step_size": 0.000001
  }


Data Input
..........
This section specifies the input size into the network. It is
required because the node list does not contain any information
on input or output shapes of the nodes.

.. include:: layertypes.rst
