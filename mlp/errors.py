# -*- coding: utf-8 -*-
"""Error functions.

This module defines error functions, with the aim of model training being to
minimise the error function given a set of inputs and target outputs.

The error functions will typically measure some concept of distance between the
model outputs and target outputs, averaged over all data points in the data set
or batch.
"""

import numpy as np


class SumOfSquaredDiffsError(object):
    """Sum of squared differences (squared Euclidean distance) error."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
<<<<<<< HEAD
        NN = outputs.shape[0]
        EE = np.sum(0.5*(outputs-targets)**2) / NN
        return EE
=======
<<<<<<< HEAD
        NN = outputs.shape[0]
        EE = np.sum(0.5*(outputs-targets)**2) / NN
        return EE
=======
        raise NotImplementedError()
>>>>>>> e9657e46f5053516ab27b9fa765301fa680413f7
>>>>>>> c889c6dd8e8c6e8ec9087b6dd8e9ad56b55e1b3b

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs. This should be
            an array of shape (batch_size, output_dim).
        """
<<<<<<< HEAD
        NN = outputs.shape[0]
        grad_y = (outputs - targets) / NN
        return grad_y
=======
<<<<<<< HEAD
        NN = outputs.shape[0]
        grad_y = (outputs - targets) / NN
        return grad_y
=======
        raise NotImplementedError()
>>>>>>> e9657e46f5053516ab27b9fa765301fa680413f7
>>>>>>> c889c6dd8e8c6e8ec9087b6dd8e9ad56b55e1b3b

    def __repr__(self):
        return 'SumOfSquaredDiffsError'
