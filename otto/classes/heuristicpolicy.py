#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Definition of heuristic policies such as infotaxis."""

import numpy as np
import random
from copy import deepcopy
from .policy import Policy

# _____________________  parameters  _____________________
EPSILON = 1e-10
EPSILON_CHOICE = EPSILON
# ________________________________________________________


class HeuristicPolicy(Policy):
    """
        A heuristic policy.

        Args:
            env (SourceTracking):
                an instance of the source-tracking POMDP
            policy (int):
                    - 0: infotaxis (Vergassola, Villermaux and Shraiman, Nature 2007)
                    - 1: space-aware infotaxis
                    - 2: custom policy (to be implemented by the user)
                    - 5: random walk
                    - 6: greedy policy
                    - 7: mean distance policy
                    - 8: voting policy (Cassandra, Kaelbling & Kurien, IEEE 1996)
                    - 9: most likely state policy (Cassandra, Kaelbling & Kurien, IEEE 1996)
            steps_ahead (int, optional):
                number of anticipated future moves (default=1), > 1 only for infotaxis
            discount (float or None, optional):
                discount factor to use when steps_ahead> 1, automatically set if None (default=None)

        Attributes:
            env (SourceTracking):
                source-tracking POMDP
            policy_index (int):
                policy index
            policy_name (str):
                name of the policy
            steps_ahead (int):
                number of anticipated future moves, for infotaxis only
            discount (float or None):
                discount factor used for steps_ahead > 1, or None if steps_ahead = 1

    """

    def __init__(
            self,
            env,
            policy,
            steps_ahead=1,
            discount=None,
    ):
        super().__init__(policy=policy)  # sets policy_index and policy_name

        self.env = env
        
        if self.policy_index == 0:
            if discount is None:
                discount = 0.999
        else:
            if steps_ahead > 1:
                raise Exception("steps_ahead has to be 1 for this policy")
            if discount is not None:
                raise Exception("discount does not make sense for this policy")
            
        self.steps_ahead = int(steps_ahead)
        self.discount = discount

    def _choose_action(self, ):
        if self.policy_index == 0:
            if self.steps_ahead == 1:
                action_chosen, _ = self._infotaxis()
            elif self.steps_ahead > 1:
                if self.discount == 0.0:
                    action_chosen, _ = self._infotaxis()
                elif self.discount == 1.0:
                    action_chosen, _ = self._infotaxis_n_steps_no_discount(self.steps_ahead)
                elif 0 < self.discount < 1:
                    action_chosen, _ = self._infotaxis_n_steps(self.steps_ahead, self.discount)
                else:
                    raise Exception("discount must be between 0 and 1")
            else:
                raise Exception("steps_ahead has to be an integer larger than 1")

        elif self.policy_index == 1:
            action_chosen, _ = self._space_aware_infotaxis()

        elif self.policy_index == 2:
            action_chosen, _ = self._custom_policy()

        elif self.policy_index == 5:
            action_chosen = random.randint(0, self.env.Nactions - 1)

        elif self.policy_index == 6:
            action_chosen, _ = self._greedy_policy()

        elif self.policy_index == 7:
            action_chosen, _ = self._mean_distance_policy()

        elif self.policy_index == 8:
            action_chosen, _ = self._voting_policy()

        elif self.policy_index == 9:
            action_chosen, _ = self._most_likely_state_policy()

        else:
            raise Exception("The policy " + str(self.policy_index) + " does not exist (yet)!")

        return action_chosen

    # __ POLICY DEFINITIONS _______________________________________
    def _infotaxis(self):
        """Original infotaxis, from Vergassola, Villermaux and Shraiman (Nature 2007)"""
        delta_entropy = np.ones(self.env.Nactions) * float("inf")
        for a in range(self.env.Nactions):
            # moving agent
            agent_, move_possible = self.env._move(a, self.env.agent)
            if move_possible:
                # calculating p_end
                p_source_ = deepcopy(self.env.p_source)
                p_end = p_source_[tuple(agent_)]
                if p_end > 1 - EPSILON:
                    # force the agent to go the source (essentially never used)
                    expected_S = -EPSILON
                else:
                    # updating p_source for not finding source
                    p_source_[tuple(agent_)] = 0
                    if np.sum(p_source_) > EPSILON:
                        p_source_ /= np.sum(p_source_)

                    # extracting the evidence matrix for Bayesian inference
                    p_evidence = self.env._extract_N_from_2N(input=self.env.p_Poisson, origin=agent_)

                    # updating p_source_ by Bayesian inference
                    p_source_ = p_source_ * p_evidence
                    p_hit = np.sum(p_source_, axis=tuple(range(1, p_source_.ndim)))
                    for h in range(self.env.Nhits):
                        if p_hit[h] > EPSILON:
                            p_source_[h] /= p_hit[h]

                    # calculating entropy
                    entropy_ = self.env._entropy(p_source_,
                                             axes=tuple(range(1, p_source_.ndim)))
                    expected_S = (1.0 - p_end) * np.sum(p_hit * entropy_)

                delta_entropy[a] = expected_S - self.env.entropy

        action_chosen = np.argwhere(np.abs(delta_entropy - np.min(delta_entropy)) < EPSILON_CHOICE).flatten()[0]

        return action_chosen, -delta_entropy

    def _infotaxis_n_steps(self, steps_ahead, discount):
        """Infotaxis policy, n-steps ahead"""

        # initialization of the info using the current position
        p_source = [self.env.p_source]
        entropy = [self.env.entropy]
        backup = [0]
        agent = [np.asarray(self.env.agent)]
        p_end = [float("inf")]
        p_hit = [float("inf")]
        terminal = [0]

        # Filling the info for each step
        step = 0
        while True:
            # creating the different matrices
            p_source.append(
                np.ones([self.env.Nactions, self.env.Nhits] * (step + 1) + [self.env.N] * self.env.Ndim)
                / (self.env.N ** self.env.Ndim - 1))
            entropy.append(
                np.ones([self.env.Nactions, self.env.Nhits] * (step + 1)) * 1e16)  # large enough number
            backup.append(
                np.zeros([self.env.Nactions, self.env.Nhits] * (step + 1), dtype='float'))
            p_hit.append(np.ones([self.env.Nactions, self.env.Nhits] * (step + 1)))
            p_end.append(np.zeros([self.env.Nactions, self.env.Nhits] * (step + 1)))
            agent.append(
                -np.ones([self.env.Nactions, self.env.Nhits] * (step + 1) + [self.env.Ndim], dtype=int))
            terminal.append(np.zeros([self.env.Nactions, self.env.Nhits] * (step + 1), dtype='int'))

            for index, _ in np.ndenumerate(entropy[step]):
                if step > 0 and terminal[step][index] > 0:  # previous state was terminal: special handling
                    for a in range(self.env.Nactions):
                        for h in range(self.env.Nhits):
                            terminal[step + 1][index + (a, h)] = terminal[step][index] + 1
                            agent[step + 1][index + (a, h)] = agent[step][index].tolist()
                            p_end[step + 1][index + (a, h)] = 0.0
                            p_hit[step + 1][index + (a, h)] = 1.0 / self.env.Nhits
                            p_source[step + 1][index + (a, h)] = np.nan
                            entropy[step + 1][index + (a, h)] = 0.0
                else:
                    for a in range(self.env.Nactions):
                        # moving agent
                        agent_, move_possible = self.env._move(
                            a, agent[step][index].tolist()
                        )
                        if move_possible and not np.all(agent[step][index] == -1):
                            # calculating p_end
                            p_source_ = deepcopy(p_source[step][index])
                            p_end_ = p_source_[tuple(agent_)]

                            if p_end_ > 1 - EPSILON:
                                for h in range(self.env.Nhits):
                                    terminal[step + 1][index + (a, h)] = 1

                            # updating p_source for not finding source
                            p_source_[tuple(agent_)] = 0
                            if np.sum(p_source_) > EPSILON:
                                p_source_ /= np.sum(p_source_)

                            # extracting the evidence matrix for Bayesian inference
                            p_evidence = self.env._extract_N_from_2N(input=self.env.p_Poisson, origin=agent_)

                            # calculating the Bayesian inference
                            p_source_ = p_source_ * p_evidence
                            p_hit_ = np.sum(p_source_,
                                            axis=tuple(range(1, p_source_.ndim)))

                            # Filling the different matrices
                            for h in range(self.env.Nhits):
                                if p_hit_[h] > EPSILON:
                                    p_source_[h] /= p_hit_[h]
                                agent[step + 1][index + (a, h)] = agent_
                                p_end[step + 1][index + (a, h)] = p_end_
                                p_hit[step + 1][index + (a, h)] = p_hit_[h]
                                p_source[step + 1][index + (a, h)] = p_source_[h]
                                entropy[step + 1][index + (a, h)] = self.env._entropy(
                                    p_source_[h]
                                )

            step += 1
            if step == steps_ahead:
                # Calculating the expected cum gain for each path (maximized wrt actions)
                for s in range(steps_ahead - 1, -1, -1):
                    cum_gain_to_reduce = (
                            (1.0 - p_end[s + 1]) * p_hit[s + 1] *
                            (- entropy[s + 1] + (discount ** (s + 1)) * backup[s + 1])
                    )
                    cum_gain_reduced_over_obs = np.sum(cum_gain_to_reduce, axis=cum_gain_to_reduce.ndim - 1)
                    if s == 0:
                        action_chosen = np.argwhere(np.abs(
                            cum_gain_reduced_over_obs - np.max(cum_gain_reduced_over_obs)) < EPSILON_CHOICE).flatten()[
                            0]
                    else:
                        backup[s] = entropy[s] + np.amax(cum_gain_reduced_over_obs,
                                                         axis=cum_gain_reduced_over_obs.ndim - 1)
                break

        return action_chosen, self.env.entropy + cum_gain_reduced_over_obs

    def _infotaxis_n_steps_no_discount(self, steps_ahead):
        """Infotaxis policy, n-steps ahead, without any discounting"""

        # initialization of the info using the current position
        p_source = [self.env.p_source]
        entropy = [self.env.entropy]
        agent = [np.asarray(self.env.agent)]
        p_end = [float("inf")]
        p_hit = [float("inf")]
        terminal = [0]

        # Filling the info for each step
        step = 0
        while True:
            # creating the different matrices
            p_source.append(
                np.ones([self.env.Nactions, self.env.Nhits] * (step + 1) + [self.env.N] * self.env.Ndim)
                / (self.env.N ** self.env.Ndim - 1))
            entropy.append(
                np.ones([self.env.Nactions, self.env.Nhits] * (step + 1)) * float("inf"))
            p_hit.append(np.ones([self.env.Nactions, self.env.Nhits] * (step + 1)))
            p_end.append(np.zeros([self.env.Nactions, self.env.Nhits] * (step + 1)))
            agent.append(
                -np.ones([self.env.Nactions, self.env.Nhits] * (step + 1) + [self.env.Ndim], dtype=int))
            terminal.append(np.zeros([self.env.Nactions, self.env.Nhits] * (step + 1), dtype='int'))

            for index, _ in np.ndenumerate(entropy[step]):
                if step > 0 and terminal[step][index] > 0:  # previous state was terminal: special handling
                    for a in range(self.env.Nactions):
                        for h in range(self.env.Nhits):
                            terminal[step + 1][index + (a, h)] = terminal[step][index] + 1
                            agent[step + 1][index + (a, h)] = agent[step][index].tolist()
                            p_end[step + 1][index + (a, h)] = 0.0
                            p_hit[step + 1][index + (a, h)] = 1.0 / self.env.Nhits
                            p_source[step + 1][index + (a, h)] = np.nan
                            entropy[step + 1][index + (a, h)] = - terminal[step][index]
                else:
                    for a in range(self.env.Nactions):
                        # moving agent
                        agent_, move_possible = self.env._move(
                            a, agent[step][index].tolist()
                        )
                        if move_possible and not np.all(agent[step][index] == -1):
                            # calculating p_end
                            p_source_ = deepcopy(p_source[step][index])
                            p_end_ = p_source_[tuple(agent_)]

                            if p_end_ > 1 - EPSILON:
                                for h in range(self.env.Nhits):
                                    terminal[step + 1][index + (a, h)] = 1

                            # updating p_source for not finding source
                            p_source_[tuple(agent_)] = 0
                            if np.sum(p_source_) > EPSILON:
                                p_source_ /= np.sum(p_source_)

                            # extracting the evidence matrix for Bayesian inference
                            p_evidence = self.env._extract_N_from_2N(input=self.env.p_Poisson, origin=agent_)

                            # calculating the Bayesian inference
                            p_source_ = p_source_ * p_evidence
                            p_hit_ = np.sum(p_source_,
                                            axis=tuple(range(1, p_source_.ndim)))

                            # Filling the different matrices
                            for h in range(self.env.Nhits):
                                if p_hit_[h] > EPSILON:
                                    p_source_[h] /= p_hit_[h]
                                agent[step + 1][index + (a, h)] = agent_
                                p_end[step + 1][index + (a, h)] = p_end_
                                p_hit[step + 1][index + (a, h)] = p_hit_[h]
                                p_source[step + 1][index + (a, h)] = p_source_[h]
                                entropy[step + 1][index + (a, h)] = self.env._entropy(
                                    p_source_[h]
                                )

            step += 1

            if step == steps_ahead:
                # Calculating the expected S for each path (minimising wrt actions)
                for s in range(steps_ahead - 1, -1, -1):
                    expected_S = (
                            (1.0 - p_end[s + 1]) * p_hit[s + 1] * entropy[s + 1]
                    )
                    expected_S = np.sum(expected_S, axis=expected_S.ndim - 1)
                    if s == 0:
                        action_chosen = np.argwhere(np.abs(expected_S - np.min(expected_S)) < EPSILON_CHOICE).flatten()[
                            0]
                    else:
                        entropy[s] = np.amin(expected_S, axis=expected_S.ndim - 1)

                break

        return action_chosen, self.env.entropy - expected_S

    def _init_space_aware_infotaxis(self, ):
        size = 1 + 2 * self.env.N
        origin = [self.env.N] * self.env.Ndim
        self.distance_array = self.env._distance(N=size, origin=origin, norm="Manhattan")

    def _space_aware_infotaxis(self, ):
        """Policy that minimizes an empirical proxi of the value function based on the entropy and the mean distance."""

        if not hasattr(self, 'distance_array'):
            # assumes Manhattan norm for distances, this can be changed in _init_space_aware_infotaxis()
            self._init_space_aware_infotaxis()

        to_minimize = np.ones(self.env.Nactions) * float("inf")
        for a in range(self.env.Nactions):
            # moving agent
            agent_, move_possible = self.env._move(a, self.env.agent)
            if move_possible:
                # array of Manhattan distances
                dist = self.env._extract_N_from_2N(input=self.distance_array, origin=agent_)

                # calculating p_end
                p_source_ = deepcopy(self.env.p_source)
                p_end = p_source_[tuple(agent_)]
                if p_end > 1 - EPSILON:
                    # force the agent to go the source (essentially never used)
                    expected_value = - EPSILON
                else:
                    # updating p_source for not finding source
                    p_source_[tuple(agent_)] = 0
                    if np.sum(p_source_) > EPSILON:
                        p_source_ /= np.sum(p_source_)

                    # extracting the evidence matrix for Bayesian inference
                    p_evidence = self.env._extract_N_from_2N(input=self.env.p_Poisson, origin=agent_)

                    # updating p_source_ by Bayesian inference
                    p_source_ = p_source_ * p_evidence
                    p_hit = np.sum(p_source_, axis=tuple(range(1, p_source_.ndim)))
                    for h in range(self.env.Nhits):
                        if p_hit[h] > EPSILON:
                            p_source_[h] /= p_hit[h]

                    # estimating expected time
                    expected_value = 0.0
                    for h in range(self.env.Nhits):
                        D = np.sum(p_source_[h] * dist)
                        H = self.env._entropy(p_source_[h])
                        value = D + 2 ** (H - 1) - 1 / 2
                        if value > 0.0:
                            value = np.log2(value)
                        expected_value += (1.0 - p_end) * p_hit[h] * value

                to_minimize[a] = expected_value

        action_chosen = np.argwhere(np.abs(to_minimize - np.min(to_minimize)) < EPSILON_CHOICE).flatten()[0]

        return action_chosen, to_minimize

    def _greedy_policy(self):
        """Usual greedy policy"""
        p = np.ones(self.env.Nactions) * float("inf")
        for a in range(self.env.Nactions):
            agent_, move_possible = self.env._move(a, self.env.agent)
            if move_possible:
                p[a] = 1.0 - self.env.p_source[tuple(agent_)]
        action_chosen = np.argwhere(np.abs(p - np.min(p)) < EPSILON_CHOICE).flatten()[0]
        return action_chosen, p

    def _init_mean_distance_policy(self, ):
        size = 1 + 2 * self.env.N
        origin = [self.env.N] * self.env.Ndim
        self.distance_array = self.env._distance(N=size, origin=origin, norm="Manhattan")

    def _mean_distance_policy(self, ):
        """Policy that chooses the action that minimizes the expected distance to the source at the next step"""

        if not hasattr(self, 'distance_array'):
            self._init_mean_distance_policy()

        to_minimize = np.ones(self.env.Nactions) * float("inf")
        for a in range(self.env.Nactions):
            # moving agent
            agent_, move_possible = self.env._move(a, self.env.agent)
            if move_possible:
                # calculating p_end
                p_source_ = deepcopy(self.env.p_source)
                p_end = p_source_[tuple(agent_)]
                if p_end > 1 - EPSILON:
                    # force the agent to go the source (essentially never used)
                    to_minimize[a] = -EPSILON
                else:
                    # updating p_source for not finding source
                    p_source_[tuple(agent_)] = 0
                    if np.sum(p_source_) > EPSILON:
                        p_source_ /= np.sum(p_source_)

                    # extracting the evidence matrix for Bayesian inference
                    p_evidence = self.env._extract_N_from_2N(input=self.env.p_Poisson, origin=agent_)

                    # updating p_source_ by Bayesian inference
                    p_source_ = p_source_ * p_evidence
                    p_hit = np.sum(p_source_, axis=tuple(range(1, p_source_.ndim)))
                    for h in range(self.env.Nhits):
                        if p_hit[h] > EPSILON:
                            p_source_[h] /= p_hit[h]

                    # calculating the distance term
                    D = self.env._extract_N_from_2N(input=self.distance_array, origin=agent_)
                    D = np.sum(p_source_ * D, axis=tuple(range(1, p_source_.ndim)))

                    # minimize a linear combination of the two
                    to_minimize[a] = (1.0 - p_end) * np.sum(p_hit * D)

        action_chosen = np.argwhere(np.abs(to_minimize - np.min(to_minimize)) < EPSILON_CHOICE).flatten()[0]

        return action_chosen, to_minimize

    def _voting_policy(self, ):
        """Policy that chooses the action which is the most likely to be optimal,
        from Cassandra, Kaelbling & Kurien (IEEE 1996)"""

        # Creating meshgrid relative to agent's position
        X = np.mgrid[tuple([range(self.env.N)] * self.env.Ndim)]
        agent = np.array(self.env.agent)
        X -= agent.reshape(tuple([self.env.Ndim] + [1] * self.env.Ndim))

        # Creating masks
        mask = np.full(tuple([self.env.Nactions] +
                             [self.env.N] * self.env.Ndim), False)

        for action in range(self.env.Nactions):
            axis = action // 2
            direction = 2 * (action % 2) - 1
            d_normal = np.linalg.norm(X[np.arange(len(X)) != axis], axis=0)
            mask[action] = direction * X[axis] >= d_normal  # quadrants mask, includes diagonals

        # Compute the p_source associated to each mask, the sum is > 1 since the diagonals are counted several times
        to_maximize = np.ones(self.env.Nactions) * float("inf")
        for a in range(self.env.Nactions):
            to_maximize[a] = np.sum(self.env.p_source[mask[a]])

        action_chosen = np.argwhere(np.abs(to_maximize - np.max(to_maximize)) < EPSILON_CHOICE).flatten()[0]

        return action_chosen, to_maximize

    def _most_likely_state_policy(self, ):
        """Policy that performs the action that is optimal for the most likely source location,
         from Cassandra, Kaelbling & Kurien (IEEE 1996)"""

        to_minimize = np.ones(self.env.Nactions) * float("inf")
        for a in range(self.env.Nactions):
            # moving agent
            agent_, move_possible = self.env._move(a, self.env.agent)
            if move_possible:
                # most likely source location
                most_likely_source = np.unravel_index(np.argmax(self.env.p_source, axis=None), self.env.p_source.shape)
                # Manhattan distance between agent and source
                to_minimize[a] = np.linalg.norm(np.asarray(agent_) - np.asarray(most_likely_source), ord=1)

        action_chosen = np.argwhere(np.abs(to_minimize - np.min(to_minimize)) < EPSILON_CHOICE).flatten()[0]

        return action_chosen, to_minimize

    def _custom_policy(self, ):
        """Implement your custom policy"""
        # implement your policy here
        # ....
        to_minimize = np.ones(self.env.Nactions)
        action_chosen = np.argwhere(np.abs(to_minimize - np.min(to_minimize)) < EPSILON_CHOICE).flatten()[0]
        return action_chosen, to_minimize
