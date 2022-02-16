#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides the ValueModel class, for defining a neural network model of the value function."""

import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers


def reload_model(model_dir, inputshape):
    """Load a model.

    Args:
        model_dir (str): path to the model
        inputshape (ndarray): shape of the neural network input (attribute of the SourceTracking env)
    """
    model_name = os.path.basename(model_dir)
    weights_path = os.path.abspath(os.path.join(model_dir, model_name))
    config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
    with open(config_path, 'rb') as filehandle:
        config = pickle.load(filehandle)
    model = ValueModel(**config)
    model.build_graph(input_shape_nobatch=inputshape)
    model.load_weights(weights_path)
    return model


class ValueModel(Model):
    """Neural network model used to predict the value of the belief state
    (i.e. the expected remaining time to find the source).

    Args:
        Ndim (int):
            number of space dimensions (1D, 2D, ...) for the search problem
        FC_layers (int):
            number of hidden layers
        FC_units (int or tuple(int)):
            units per layer
        regularization_factor (float, optional):
            factor for regularization losses (default=0.0)
        loss_function (str, optional):
            either 'mean_absolute_error', 'mean_absolute_percentage_error' or 'mean_squared_error' (default)

    Attributes:
        config (dict):
            saves the args in a dictionary, can be used to recreate the model

    """

    def __init__(self,
                 Ndim,
                 FC_layers,
                 FC_units,
                 regularization_factor=0.0,
                 loss_function='mean_squared_error',
                 ):
        """Constructor.


        """
        super(ValueModel, self).__init__()
        self.config = {"Ndim": Ndim,
                       "FC_layers": FC_layers,
                       "FC_units": FC_units,
                       "regularization_factor": regularization_factor,
                       "loss_function": loss_function,
                       }

        self.Ndim = Ndim

        if loss_function == 'mean_absolute_error':
            self.loss_function = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_function == 'mean_absolute_percentage_error':
            self.loss_function = tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_function == 'mean_squared_error':
            self.loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        else:
            raise Exception("This loss function has not been made available")

        regularizer = regularizers.l2(regularization_factor)

        # flattening
        self.flatten = layers.Flatten()

        # fully connected layers
        self.FC_block = None
        if FC_layers > 0:
            if isinstance(FC_units, int):
                FC_units = tuple([FC_units] * FC_layers)
            if len(FC_units) != FC_layers:
                raise Exception("Must provide nb of units for each dense layer or provide a single int")
            self.FC_block = []
            for i in range(FC_layers):
                dense_ = layers.Dense(
                    units=FC_units[i],
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activity_regularizer=regularizer,
                )
                self.FC_block.append(dense_)

        # last linear layer
        self.densefinal = layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1),
            kernel_constraint=tf.keras.constraints.non_neg(),
            bias_constraint=tf.keras.constraints.non_neg(),
        )

    def call(self, x, training=False, sym_avg=False):
        """Call the value model

        Args
            x (ndarray or tf.Tensor with shape (batch_size, input_shape)):
                array containing a batch of inputs
            training (bool, optional):
                whether this call is done during training (as opposed to evaluation) (default=False)
            sym_avg (bool, optional):
                whether to take the average value of symmetric duplicates (default=False)

        Returns
            x (tf.Tensor with shape (batch_size, 1))
                array containing a batch of values

        Raises
            Exception: if symmetric duplicates are not implemented for that Ndim
        """
        shape = x.shape  # (batch_size, input_shape)
        ensemble_sym_avg = False
        if sym_avg and (shape[0] is not None):
            ensemble_sym_avg = True

        # create symmetric duplicates
        if ensemble_sym_avg:
            if self.Ndim == 1:
                Nsym = 2
                x = x[tf.newaxis, ...]
                _ = tf.reverse(x, axis=[2])  # symmetry: x -> -x
                x = tf.concat([x, _], axis=0)
                x = tf.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            elif self.Ndim == 2:
                Nsym = 8
                x = x[tf.newaxis, ...]
                _ = tf.transpose(x, perm=[0, 1, 3, 2, 4])  # transposition
                x = tf.concat([x, _], axis=0)
                _ = tf.reverse(x, axis=[2])  # symmetry: x -> -x
                x = tf.concat([x, _], axis=0)
                _ = tf.reverse(x, axis=[2, 3])  # symmetry: x -> -x, y -> -y
                x = tf.concat([x, _], axis=0)
                x = tf.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            else:
                raise Exception("symmetric duplicates for Ndim > 2 is not implemented")

        # flatten input
        x = self.flatten(x)

        # forward pass
        if self.FC_block is not None:
            for i in range(len(self.FC_block)):
                x = self.FC_block[i](x, training=training)

        x = self.densefinal(x)

        # reduce the symmetric outputs
        if ensemble_sym_avg:
            x = tf.reshape(x, shape=(Nsym, shape[0], 1))
            x = tf.reduce_mean(x, axis=0)

        return x  # (batch_size, 1)

    def build_graph(self, input_shape_nobatch):
        """Builds the model. Use this function instead of model.build() so that a call to
        model.summary() gives shape information.
        """
        input_shape_nobatch = tuple(input_shape_nobatch)
        input_shape_withbatch = tuple([1] + list(input_shape_nobatch))
        self.build(input_shape_withbatch)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)

    # note: the tf.function decorator prevent using tensor.numpy() for performance reasons, use only tf operations
    @tf.function
    def train_step(self, x, y, augment=False):
        """A training step.

        Args:
            x (tf.Tensor with shape=(batch_size, input_shape)): batch of inputs
            y (tf.Tensor with shape=(batch_size, 1)): batch of target values

        Returns
            loss (tf.Tensor with shape=()): total loss
        """

        # Add symmetric duplicates
        if augment:
            shape = x.shape
            if self.Ndim == 1:
                Nsym = 2
                x = x[tf.newaxis, ...]
                _ = tf.reverse(x, axis=[2])  # symmetry: x -> -x
                x = tf.concat([x, _], axis=0)
                x = tf.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            elif self.Ndim == 2:
                Nsym = 8
                x = x[tf.newaxis, ...]
                _ = tf.transpose(x, perm=[0, 1, 3, 2, 4])  # transposition
                x = tf.concat([x, _], axis=0)
                _ = tf.reverse(x, axis=[2])  # symmetry: x -> -x
                x = tf.concat([x, _], axis=0)
                _ = tf.reverse(x, axis=[2, 3])  # symmetry: x -> -x, y -> -y
                x = tf.concat([x, _], axis=0)
                x = tf.reshape(x, shape=tuple([Nsym * shape[0]] + list(shape[1:])))
            else:
                raise Exception("augmentation with symmetric duplicates is not implemented for Ndim > 2")

            # repeat target
            y = y[tf.newaxis, ...]
            y = tf.repeat(y, Nsym, axis=0)
            y = tf.reshape(y, shape=tuple([Nsym * shape[0]] + [1]))

        # Compute predictions
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            loss_err = self.loss_function(y, y_pred)  # compute loss
            loss_reg = tf.math.reduce_sum(self.losses)  # adding the regularization losses
            loss = tf.math.add(loss_err, loss_reg)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute total loss
        loss = tf.math.reduce_mean(loss)

        return loss

    @tf.function
    def test_step(self, x, y):
        """ A test step.

        Args:
            x (tf.Tensor with shape=(batch_size, input_shape)): batch of inputs
            y (tf.Tensor with shape=(batch_size, 1)): batch of target values

        Returns
            loss (tf.Tensor with shape=()): total loss
        """

        # Compute predictions
        y_pred = self(x, training=False)

        # Compute the loss
        loss_err = self.loss_function(y, y_pred)  # compute loss
        loss_reg = tf.math.reduce_sum(self.losses)  # adding the regularization losses
        loss = tf.math.add(loss_err, loss_reg)

        # Compute total loss
        loss = tf.math.reduce_mean(loss)

        return loss

    def save_model(self, model_dir):
        """Save the model to model_dir."""
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_name = os.path.basename(model_dir)
        weights_path = os.path.abspath(os.path.join(model_dir, model_name))
        self.save_weights(weights_path, save_format='h5')
        config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
        with open(config_path, 'wb') as filehandle:
            pickle.dump(self.config, filehandle)

        