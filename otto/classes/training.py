#!/usr/bin/self python3
# -*- coding: utf-8 -*-
"""Functions and classes required to train a value model on the source-tracking POMDP."""
import os
import numpy as np
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from .sourcetracking import SourceTracking

# _____________________  parameters  _____________________
EPSILON = 1e-10
# ________________________________________________________


class State:
    """
    Defines a (belief) state. This is the class used to store transitions (s, s').

    Attributes:
        p_source (ndarray):
            probability distribution of the source
        agent (list(int)):
            location of the agent
        prob (float):
            - if current state s, then prob=1.0;
            - if next state s', then probability to transit from s to this state
    """

    def __init__(
    self,
    p_source,
    agent,
    prob=1.0,
    ):

        if prob > EPSILON:
            assert abs(np.sum(p_source) - 1.0) < EPSILON
        self.p_source = np.asarray(p_source, dtype=np.float32)
        self.agent = agent
        self.prob = prob


class TrainingEnv(SourceTracking):
    """Add functions useful for training to the SourceTracking class"""

    def __init__(
            self,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_action_from_statep(self, model, statep=None, sym_avg=False):  # statep: (Nactions, Nhits)
        """Choose an action according to the current value model and successor states statep.
        Has the same effect as choose_action from the RLPolicy class, but is faster if statep is provided.

        Args:
            model (ValueModel):
                value model to be used
            statep (ndarray or None, optional):
                array of all possible next states reachable from current state, as computed using :func:`transitions`;
                if None then will be computed within this function (default=None)
            sym_avg (bool, optional):
                whether to average the value over symmetric-equivalent states (default=False)

        Returns:
            action_chosen (int): action chosen according to the policy
        """
        if statep is None:
            _, statep = self.transitions()
        action_chosen, _ = self._value_policy_from_statep(model=model, statep=statep, sym_avg=sym_avg)

        return action_chosen

    def transitions(self):
        """ Compute all possible successors s' that can be reached from state s

        Returns:
            state (State): current state
            statep (ndarray of State): array of all states reached for all possible actions and hit values
        """
        statep = [0] * self.Nactions

        for action in range(self.Nactions):
            statep[action] = []

            # moving agent
            agent_, move_possible = self._move(action, self.agent)

            # extracting the evidence matrix for Bayesian inference
            p_evidence = self._extract_N_from_2N(input=self.p_Poisson, origin=agent_)

            # updating p_source by Bayesian inference
            p_source_ = deepcopy(self.p_source)
            p_source_ = p_source_[np.newaxis, :] * p_evidence

            for h in range(self.Nhits):
                prob = np.sum(p_source_[h])  # norm is the proba to transit to this state = (1-pend) * p(hit)
                prob = np.maximum(EPSILON, prob)
                state_ = State(p_source_[h] / prob, agent_, prob=prob)
                statep[action].append(state_)

        state = State(self.p_source, self.agent, prob=1.0)

        statep = np.asarray(statep)  # (Nactions, Nhits)

        return state, statep

    def states2inputs(self, states, dims):
        """Convert states into inputs required to compute their value with the model.

        Args:
            states (State object or ndarray of State objects):
                - if current states s: single State with no shape or array with shape (batch_size, )
                - if next states s': array with shape (Nactions, Nhits) or shape (batch_size, Nactions, Nhits)
            dims (0 or 2):
                flag used to differentiate between current states and next states
                - if current states s: 0
                - if next states s': 2 (this is because one current state yields (Nactions, Nhits) next states)

        Returns:
            inputs (ndarray):
                array of inputs for the model, with:
                - if current states s: shape (batch_size, input_shape)
                - if next states s': shape (batch_size, Nactions, Nhits, input_shape)
            prob (ndarray):
                - if current states s: array of 1.0, with shape (batch_size, )
                - if next states s': array of transition probabilities, with shape (batch_size, Nactions, Nhits)
        """
        # if input is a single state: numpyfy
        if dims == 0 and isinstance(states, type(State(1, 0, 0))):
            states = np.array(states)  # ()

        if dims == 0:
            # if leading dim (batch_size) is missing, add one
            if states.ndim == 0:
                states = states[np.newaxis]  # (1,)
            # check shape
            assert states.ndim == 1
        elif dims == 2:
            # if leading dim (batch_size) is missing, add one
            if states.ndim == 2:
                states = states[np.newaxis, :]  # (1, Nactions, Nhits)
            # check shape
            assert states.ndim == 3
            assert states.shape[1:] == (self.Nactions, self.Nhits)
        else:
            raise Exception("ndim must be 0 (for state) or 2 (for statep)")

        ishape = tuple(list(states.shape) + list(self.NN_input_shape))
        zshape = tuple(list(states.shape))

        states = states.ravel()

        inputs = []
        probs = []
        for state in states:
            inputs.append(self._state2input(state))
            probs.append(state.prob)

        inputs = np.asarray(inputs, dtype=np.float32)
        inputs = np.reshape(inputs, ishape)

        probs = np.asarray(probs, dtype=np.float32)
        probs = np.reshape(probs, zshape)

        return inputs, probs

    def get_state_value(self, model, state):
        """
        Returns the value of a current state according to the model

        Args:
            model (ValueModel): model to be used
            state (State or ndarray): single State object or numpy array of State objects, with shape (batch_size,)

        Returns:
            value (ndarray): numpy array of values, with shape (batch_size, 1)
        """
        input, _ = self.states2inputs(state, dims=0)
        value = model(input)
        return value.numpy()

    def get_target(self, modelvalue, modelaction, statep):
        """
        Compute the target value (for training) of a state s.

        Args:
            modelvalue (ValueModel):
                model used to compute values
            modelaction (ValueModel or None):
                model used to choose action, if None then same as modelvalue
            statep (ndarray):
                array of States objects with shape (Nactions, Nhits) or (batch_size, Nactions, Nhits),
                containing all states s' reached from state s for all possible actions and hit values

        Returns:
            target (ndarray): array of target values with shape (batch_size, 1)

        """
        # statep: numpy array of States: (Nactions, Nhits) or (batch_size, Nactions, Nhits)
        # target: (batch_size, 1)

        if statep.ndim == 2:  # must add batch dimension
            statep = statep[np.newaxis, :]

        batch_size = statep.shape[0]

        ishape = self.NN_input_shape

        inputp, probp = self.states2inputs(statep, dims=2)

        assert inputp.shape == tuple([batch_size] + [self.Nactions] + [self.Nhits] + list(ishape))
        assert probp.shape == tuple([batch_size] + [self.Nactions] + [self.Nhits])

        # arranging shapes
        inputpshape = inputp.shape
        inputp = tf.reshape(inputp,
                            shape=tuple([inputpshape[0] * inputpshape[1] * inputpshape[2]] + list(inputpshape[3:])))
        assert inputp.shape == tuple([batch_size * self.Nactions * self.Nhits] + list(ishape))

        # calculating the target
        values_next = modelvalue(inputp)
        if modelaction is not None:
            values_next_for_action = modelaction(inputp)

        assert values_next.shape == tuple([batch_size * self.Nactions * self.Nhits] + [1])
        if modelaction is not None:
            assert values_next_for_action.shape == tuple([batch_size * self.Nactions * self.Nhits] + [1])

        values_next = tf.reshape(values_next,
                                 shape=(inputpshape[0], inputpshape[1], inputpshape[2],))
        assert values_next.shape == tuple([batch_size] + [self.Nactions] + [self.Nhits])
        if modelaction is not None:
            values_next_for_action = tf.reshape(values_next_for_action,
                                                shape=(inputpshape[0], inputpshape[1], inputpshape[2],))
            assert values_next_for_action.shape == tuple([batch_size] + [self.Nactions] + [self.Nhits])

        assert values_next.shape == probp.shape
        if modelaction is not None:
            assert values_next_for_action.shape == probp.shape

        # sum over hits
        expected_value = 1.0 + tf.math.reduce_sum(probp * values_next, axis=-1)
        assert expected_value.shape == tuple([batch_size] + [self.Nactions])

        if modelaction is not None:
            expected_value_for_action = 1.0 + tf.math.reduce_sum(probp * values_next_for_action, axis=-1)
            assert expected_value_for_action.shape == tuple([batch_size] + [self.Nactions])

        # reduce over best action
        if modelaction is not None:
            # get action indices
            chosen_action = tf.math.argmin(expected_value_for_action, axis=1).numpy()  # (batch_size,)
            # get array of indices as required for gather_nd
            ind = np.stack((np.arange(batch_size, dtype='int'), chosen_action)).transpose()  # (batch_size, 2)
            # get expected_value[action_indices]
            target = tf.gather_nd(expected_value, ind)  # (batch_size,)
            target = target[..., tf.newaxis]  # (batch_size, 1)
        else:
            # min over actions
            target = tf.math.reduce_min(expected_value, axis=1, keepdims=True)  # (batch_size, 1)

        assert target.shape == tuple([batch_size] + [1])

        return target.numpy()

    def apply_sym_transformation(self, sym):
        """
        Apply a symmetry transformation (rotation, mirror, flip, etc.) to p_source, agent, hit_map, ...

        Args:
            sym (int): which symmetry transformation to apply (0 for none)

        """
        self.p_source = self._sym_transformation_array(self.p_source, sym=sym)
        self.hit_map = self._sym_transformation_array(self.hit_map, sym=sym)
        self.agent = self._sym_transformation_coords(self.agent, sym=sym)
        self._agento = self._sym_transformation_coords(self._agento, sym=sym)
        self._agentoo = self._sym_transformation_coords(self._agentoo, sym=sym)
        if self.draw_source:
            self.source = self._sym_transformation_coords(self.source, sym=sym)

    # __ INTERNAL FUNCTIONS  _________________________________________
    def _value_policy_from_statep(self, model, statep, sym_avg=False):  # statep: (Nactions, Nhits)
        assert statep.shape == (self.Nactions, self.Nhits)

        ishape = self.NN_input_shape

        inputp, probp = self.states2inputs(statep, dims=2)  # (1, Nactions, Nhits, inputshape), (1, Nactions, Nhits)
        assert inputp.shape == tuple([1] + [self.Nactions] + [self.Nhits] + list(ishape))
        assert probp.shape == tuple([1] + [self.Nactions] + [self.Nhits])

        inputpshape = inputp.shape
        inputp = tf.reshape(inputp, shape=tuple([inputpshape[0] * inputpshape[1] * inputpshape[2]] + list(inputpshape[3:])))  # (1 * Nactions * Nhits, inputshape)
        assert inputp.shape == tuple([1 * self.Nactions * self.Nhits] + list(ishape))

        values_next = model(inputp, sym_avg=sym_avg)  # (1 * Nactions*Nhits, 1)
        assert values_next.shape == tuple([1 * self.Nactions * self.Nhits] + [1])

        values_next = tf.reshape(values_next, shape=(inputpshape[0], inputpshape[1], inputpshape[2],))  # (1, Nactions, Nhits)
        assert values_next.shape == tuple([1] + [self.Nactions] + [self.Nhits])
        assert values_next.shape == probp.shape

        expected_value = 1.0 + tf.math.reduce_sum(probp * values_next, axis=2)  # sum over hits: (1, Nactions)
        assert expected_value.shape == tuple([1] + [self.Nactions])

        action_chosen = np.argmin(expected_value.numpy().squeeze())
        return action_chosen, expected_value.numpy().squeeze()

    def _state2input(self, state):
        p_source = self._centeragent(state.p_source, state.agent)
        input = np.array(p_source, dtype=np.float32)
        assert input.shape == self.NN_input_shape
        return input

    def _sym_transformation_array(self, x, sym):
        # x =  numpy array
        if x.ndim == 1:
            if not 0 <= sym <= 1:
                raise Exception("sym must be 0 or 1")
            if sym == 1:
                x = np.flip(x, axis=0)  # x -> -x
        elif x.ndim == 2:
            if not 0 <= sym <= 7:
                raise Exception("sym must be between 0 and 7")
            if sym == 1 or sym > 4:
                x = np.transpose(x, axes=(1, 0))  # x -> y
            if sym == 2 or sym == 5:
                x = np.flip(x, axis=0)  # x -> -x
            if sym == 3 or sym == 6:
                x = np.flip(x, axis=1)  # y -> -y
            if sym == 4 or sym == 7:
                x = np.flip(x, axis=(0, 1))  # x -> -x, y -> -y
        else:
            raise Exception("_sym_transformation_array is not implemented for Ndim > 2")
        return x

    def _sym_transformation_coords(self, x, sym):
        # x = [x1, x2, ...]
        x = np.array(x)
        if len(x) == 1:
            if not 0 <= sym <= 1:
                raise Exception("sym must be 0 or 1")
            if sym == 1:
                x[0] = self.N - 1 - x[0]  # x -> -x
        elif len(x) == 2:
            if not 0 <= sym <= 7:
                raise Exception("sym must be between 0 and 7")
            if sym == 1 or sym > 4:
                x = np.flip(x)  # x -> y
            if sym == 2 or sym == 5:
                x[0] = self.N - 1 - x[0]  # x -> -x
            if sym == 3 or sym == 6:
                x[1] = self.N - 1 - x[1]  # y -> -y
            if sym == 4 or sym == 7:
                x[0] = self.N - 1 - x[0]  # x -> -x
                x[1] = self.N - 1 - x[1]  # y -> -y
        else:
            raise Exception("_sym_transformation_coords is not implemented for Ndim > 2")
        x = x.tolist()
        return x
