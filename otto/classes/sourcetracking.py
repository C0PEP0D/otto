#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides the SourceTracking class, to simulate the source-tracking POMDP."""

import numpy as np
import warnings
from copy import deepcopy
from scipy.special import kv
from scipy.special import kn
from scipy.special import gamma as Gamma
from scipy.stats import poisson as Poisson_distribution

# _____________________  parameters  _____________________
EPSILON = 1e-10
# ________________________________________________________


class SourceTracking:
    """Environment used to simulate the source-tracking POMDP.

    Args:
        Ndim (int):
            number of space dimensions (1D, 2D...)
        lambda_over_dx (float):
            dimensionless problem size (odor dispersion lengthscale divided by agent step size)
        R_dt (float):
            dimensionless source intensity (source rate of emission multiplied by the agent time step)
        norm_Poisson ('Euclidean', 'Manhattan' or 'Chebyshev', optional):
            norm used for hit detections (default='Euclidean')
        Ngrid (int or None, optional):
            linear size of the domain, set automatically if None (default=None)
        Nhits (int or None, optional):
            number of possible hit values, set automatically if None (default=None)
        draw_source (bool, optional):
            whether to actually draw the source location (otherwise uses Bayesian framework) (default=False)
        initial_hit (int or None, optional):
                value of the initial hit, if None drawn randomly according to relevant probability distribution (default=None)
        dummy (bool, optional):
                set automatic parameters (e.g., Ngrid) but does not initialize the POMDP (default=False)

    Attributes:
        Ndim (int):
            number of dimension of space (1D, 2D...)
        lambda_over_dx (float):
            dimensionless problem size (odor dispersion lengthscale divided by agent step size)
        R_dt (float):
            dimensionless source intensity (source rate of emission multiplied by the agent time step)
        norm_Poisson (str):
            norm used for hit detections: 'Euclidean', 'Manhattan' or 'Chebyshev'
        Ngrid (int):
            linear size of the domain
        Nhits (int):
            number of possible hit values
        draw_source (bool):
            whether a source location is actually drawn  (otherwise uses Bayesian framework)
        initial_hit (int):
            value of the initial hit
        Nactions (int):
            number of possible actions
        mu0_Poisson (float):
            mean number of hits at a distance lambda_over_dx from the source
        agent (list(int)):
            current agent location
        p_source (ndarray):
            current probability distribution of the source location)
        obs (dict):
            current observation ("hit" and "done")
        hit_map (ndarray):
            number of hits received for each location (-1 for cells not visited ye
        cumulative_hits (int):
            cumulated sum of hits received (ignoring initial hit)
        agent_near_boundaries (bool):
            whether the agent is currently near a boundary
        agent_stuck (bool):
            whether the agent is currently stuck in an "infinite" loop



    """

    def __init__(
        self,
        Ndim,
        lambda_over_dx,
        R_dt,
        norm_Poisson='Euclidean',
        Ngrid=None,
        Nhits=None,
        draw_source=False,
        initial_hit=None,
        dummy=False,
    ):

        self.Ndim = int(Ndim)
        if self.Ndim < 1:
            raise Exception("Ndim must be a positive integer")

        self.lambda_over_dx = lambda_over_dx
        self.R_dt = R_dt
        if self.lambda_over_dx < 0 or self.R_dt < 0:
            raise Exception("lambda_over_dx and R_dt must be > 0")
        if self.lambda_over_dx < 1:
            warnings.warn("lambda_over_dx should be >= 1 for the problem to make sense")

        self.norm_Poisson = norm_Poisson
        if not (self.norm_Poisson in ('Euclidean', 'Manhattan', 'Chebyshev')):
            raise Exception("norm_Poisson must be 'Euclidean', 'Manhattan' or 'Chebyshev'")

        self._set_mu0_Poisson()

        if Nhits is None:
            self.Nhits = self._autoset_Nhits()
        else:
            self.Nhits = int(Nhits)
        if self.Nhits < 2:
            raise Exception("Nhits must be at least 2")

        if Ngrid is None:
            self.N = self._autoset_Ngrid()
        else:
            self.N = (Ngrid // 2) * 2 + 1   # make it odd
        if self.N < 3:
            raise Exception("Ngrid must be at least 3")

        self.draw_source = draw_source
        if not isinstance(self.draw_source, bool):
            raise Exception("draw_source must be a bool")

        self.Nactions = 2 * self.Ndim

        if not dummy:
            self.restart(initial_hit)

    def restart(self, initial_hit=None):
        """Restart the search.

        Args:
            initial_hit (int or None): initial hit, if None then random
        """
        if initial_hit is None:
            self.initial_hit = self._initial_hit()
        else:
            self.initial_hit = int(initial_hit)
        if self.initial_hit > self.Nhits - 1:
            raise Exception("initial_hit cannot be > Nhits - 1")

        self.hit_map = -np.ones([self.N] * self.Ndim, dtype=int)
        self.agent = [self.N // 2] * self.Ndim
        self._init_distributed_source()
        if self.draw_source:
            self._draw_a_source()

        self.cumulative_hits = 0
        self.agent_near_boundaries = 0
        self.agent_stuck = False
        self.obs = {"hit": self.initial_hit, "done": False}

        self._agento = [0] * self.Ndim  # position 1 step ago (arbitrary init value)
        self._agentoo = [self.N] * self.Ndim  # position 2 steps ago (arbitrary init value)
        self._repeated_visits = 0  # to detect back and forth motion

    def step(self, action, hit=None, quiet=False):
        """Make a step in the source-tracking environment:
         - The agent moves to its new position according to `action`,
         - The agent receives an observation (hit),
         - The belief (self.p_source) and the hit map (self.hit_map) are updated.

        Args:
            action (int): action of the agent
            hit (int, optional): prescribed number of hits,
                if None (default) the number of hits is chosen randomly according to its probability
            quiet (bool, optional): whether to print when agent is attempting a forbidden move (default=False)

        Returns:
            hit (int): number of hits received
            p_end (float): probability of having found the source (relevant only if not draw_source)
            done (bool): whether the source has been found (relevant only if draw_source)

        """
        hit, p_end, done = self._execute_action(action, hit, quiet)
        self._update_after_hit(hit, done)

        return hit, p_end, done

    # __ POMDP UPDATES _______________________________________
    def _execute_action(self, action, hit=None, quiet=False):
        self._agentoo = self._agento
        self._agento = self.agent

        # move agent
        self.agent, is_move_possible = self._move(action, self.agent)
        if (not is_move_possible) and (not quiet):
            print("This move is not possible: agent =", self.agent, "cannot do action =",  action)
        self.agent_near_boundaries = self._is_agent_near_boundaries(n_boundary=1)
        self.agent_stuck = self._is_agent_stuck()

        if self.draw_source:
            if self.norm_Poisson == 'Manhattan':
                ord = 1
            elif self.norm_Poisson == 'Euclidean':
                ord = 2
            elif self.norm_Poisson == 'Chebyshev':
                ord = float("inf")
            else:
                raise Exception("This norm is not implemented")
            d = np.linalg.norm(np.asarray(self.agent) - np.asarray(self.source), ord=ord)

            if d > EPSILON:
                done = False
                p_end = 0

                # Picking randomly the number of hits
                mu = self._mean_number_of_hits(d)
                probability = np.zeros(self.Nhits)
                sum_proba = 0
                for h in range(self.Nhits - 1):
                    probability[h] = self._Poisson(mu, h)
                    sum_proba += self._Poisson(mu, h)
                probability[self.Nhits - 1] = np.maximum(0, 1.0 - sum_proba)
                if hit is None:
                    hit = np.random.RandomState().choice(
                        range(self.Nhits), p=probability
                    )
            else:
                done = True
                p_end = 1
                hit = - 1

        else:
            done = False

            p_end = self.p_source[tuple(self.agent)]
            if p_end > 1 - EPSILON:
                done = True

            # Source not in self.agent
            new_p_source = deepcopy(self.p_source)
            new_p_source[tuple(self.agent)] = 0.0
            if np.sum(new_p_source) > EPSILON:
                new_p_source /= np.sum(new_p_source)
            else:
                done = True

            if not done:
                # extracting the evidence matrix for Bayesian inference
                p_evidence = self._extract_N_from_2N(input=self.p_Poisson, origin=self.agent)

                # Compute hit proba
                p_hit_table = np.zeros(self.Nhits)
                for h in range(self.Nhits):
                    p_hit_table[h] = np.maximum(
                        0,
                        np.sum(new_p_source * p_evidence[h])
                    )
                sum_p_hit = np.sum(p_hit_table)
                if np.abs(sum_p_hit - 1.0) < EPSILON:
                    p_hit_table /= sum_p_hit
                else:
                    print("sum_p_hit_table = ", sum_p_hit)
                    raise Exception("p_hit_table does not sum to 1")

                # Picking randomly the number of hits
                if hit is None:
                    hit = np.random.RandomState().choice(range(self.Nhits), p=p_hit_table)

            else:
                hit = self.Nhits - 1

        if not done:
            self.cumulative_hits += hit

        return hit, p_end, done
    
    def _update_after_hit(self, hit, done=None):
        """Update of the hit_map and p_source when receiving hits.

        Args:
            hit (int): number of hits received
            done (bool): whether the unique source is found
        """
        if hit is not None:
            self._update_hit_map(hit)
            self._update_obs(hit, done)
            self._update_p_source(hit, done)
            
    def _update_hit_map(self, hit=0):
        self.hit_map[tuple(self.agent)] = hit

    def _update_obs(self, hit, done):
        self.obs["hit"] = hit
        self.obs["done"] = done

    def _update_p_source(self, hit=0, done=None):
        if done:
            self.p_source = np.zeros([self.N] * self.Ndim)
            self.p_source[tuple(self.agent)] = 1.0
            self.entropy = 0.0
        else:
            self.p_source[tuple(self.agent)] = 0
            p_evidence = self._extract_N_from_2N(input=self.p_Poisson, origin=self.agent)
            self.p_source *= p_evidence[hit]
            self.p_source[(self.p_source < 0.0) & (self.p_source > -1e-15)] = 0.0

            if np.sum(self.p_source) > EPSILON:
                self.p_source /= np.sum(self.p_source)
            self.entropy = self._entropy(self.p_source)

    def _move(self, action, agent):
        """Move the agent according to action.

        Args:
            action (int): action chosen
            agent (list of int): position of the agent

        Returns:
            new_agent (list of int): new position of the agent
            is_move_possible (bool): whether the action was allowed

        """
        is_move_possible = True
        new_agent = deepcopy(agent)
        axis = action // 2
        if axis < self.Ndim:
            direction = 2 * (action % 2) - 1
            if direction == -1:
                if agent[axis] > 0:
                    new_agent[axis] -= 1
                else:
                    is_move_possible = False
            elif direction == 1:
                if agent[axis] < self.N - 1:
                    new_agent[axis] += 1
                else:
                    is_move_possible = False
        else:
            raise Exception("This action is outside the allowed range")

        return new_agent, is_move_possible
    
    # __ HIT DETECTION _______________________________________
    def _mean_number_of_hits(self, distance):
        distance = np.array(distance)
        distance[distance == 0] = 1.0
        if self.Ndim == 1:
            mu = np.exp(-distance / self.lambda_over_dx + 1)
        elif self.Ndim == 2:
            mu = kn(0, distance / self.lambda_over_dx) / kn(0, 1)
        elif self.Ndim == 3:
            mu = self.lambda_over_dx / distance * np.exp(-distance / self.lambda_over_dx + 1)
        elif self.Ndim > 3:
            mu = (self.lambda_over_dx / distance) ** (self.Ndim / 2 - 1) \
                 * kv(self.Ndim / 2 - 1, distance / self.lambda_over_dx) \
                 / kv(self.Ndim / 2 - 1, 1)
        else:
            raise Exception("Problem with the number of dimensions")
        mu *= self.mu0_Poisson
        return mu

    def _Poisson_unbounded(self, mu, h):
        p = Poisson_distribution(mu).pmf(h)
        return p

    def _Poisson(self, mu, h):
        if h < self.Nhits - 1:   # = Poisson(mu,hit=h)
            p = self._Poisson_unbounded(mu, h)
        elif h == self.Nhits - 1:     # = Poisson(mu,hit>=h)
            sum = 0.0
            for k in range(h):
                sum += self._Poisson_unbounded(mu, k)
            p = 1 - sum
        else:
            raise Exception("h cannot be > Nhits - 1")
        return p
    
    def _compute_p_Poisson(self):
        size = 1 + 2 * self.N  # note: this could be reduced to size 2N - 1
        origin = [self.N] * self.Ndim
        d = self._distance(N=size, origin=origin, norm=self.norm_Poisson)
        mu = self._mean_number_of_hits(d)
        mu[tuple(origin)] = 0.0

        self.p_Poisson = np.zeros([self.Nhits] + [size] * self.Ndim)
        sum_proba = np.zeros([size] * self.Ndim)
        for h in range(self.Nhits):
            self.p_Poisson[h] = self._Poisson(mu, h)
            sum_proba += self.p_Poisson[h]
            if h < self.Nhits - 1:
                sum_is_one = np.all(abs(sum_proba - 1) < EPSILON)
                if sum_is_one:
                    raise Exception(str('Nhits is too large, reduce it to Nhits = ' + str(h + 1)
                                        + ' or lower (higher values have zero probabilities)'))

        if not np.all(sum_proba == 1.0):
            raise Exception("_compute_p_Poisson: sum proba is not 1")

        # by definition: p_Poisson(origin) = 0
        for h in range(self.Nhits):
            self.p_Poisson[tuple([h] + origin)] = 0.0
            
    # __ INITIALIZATION AND AUTOSET _______________________________________
    def _init_distributed_source(self, ):
        if not hasattr(self, 'p_Poisson'):
            self._compute_p_Poisson()
        self.p_source = np.ones([self.N] * self.Ndim) / (self.N ** self.Ndim - 1)
        self.p_source[tuple([self.N // 2] * self.Ndim)] = 0.0
        self._update_p_source(hit=self.initial_hit)
        self._update_hit_map(hit=self.initial_hit)

    def _draw_a_source(self):
        prob = self.p_source.flatten()
        index = np.random.RandomState().choice(self.N**self.Ndim, size=1, p=prob)[0]
        self.source = np.unravel_index(index, shape=([self.N] * self.Ndim))
        self.source = np.array(self.source, dtype=int)

    def _initial_hit(self, hit=None):
        if hit is None:
            p_hit_table = np.zeros(self.Nhits)
            r = np.arange(1, int(1000 * self.lambda_over_dx))
            shell_volume = self._volume_ball(r+0.5) - self._volume_ball(r-0.5)
            for h in range(1, self.Nhits):
                p = self._Poisson(self._mean_number_of_hits(r), h)  # proba hit=h as a function of distance r to the source
                p_hit_table[h] = max(0, np.sum(p * shell_volume))  # not normalized
            p_hit_table /= np.sum(p_hit_table)
            hit = np.random.RandomState().choice(range(self.Nhits), p=p_hit_table)
        return hit
    
    def _autoset_Ngrid(self, p_source_out=1e-3):
        # return the smallest Ngrid such that, for a virtually infinite domain, the probability of the source
        # being outside a ball of diameter Ngrid is less than p_source_out after any initial hit
        r = np.arange(1, int(1000 * self.lambda_over_dx))
        shell_volume = self._volume_ball(r + 0.5) - self._volume_ball(r - 0.5)
        Ngrid = 0
        hit_list = range(1, self.Nhits)
        for hit in hit_list:
            p = self._Poisson(self._mean_number_of_hits(r), hit)  # proba hit=h as a function of distance r to the source
            pball = np.cumsum(p * shell_volume)
            pball /= pball[-1]  # proba source is in the ball as a function of ball radius r
            where = np.argwhere(1 - pball < p_source_out)[0, 0]
            Ngrid = max(Ngrid, 2 * (where + 1) + 1)
        return int(Ngrid)

    def _autoset_Nhits(self):
        mu_at_dx = self._mean_number_of_hits(1.0)
        h = mu_at_dx + np.sqrt(mu_at_dx)
        return int(np.ceil(h)) + 1
    
    # __ LOW LEVEL UTILITIES _______________________________________
    def _entropy(self, array, axes=None):
        log2 = np.zeros(array.shape)
        indices = array > EPSILON
        log2[indices] = -np.log2(array[indices])
        return np.sum(array * log2, axis=axes)

    def _distance(self, Ndim=None, N=None, origin=None, norm='Euclidean'):
        if Ndim is None:
            Ndim = self.Ndim
        if N is None:
            N = self.N
        if origin is None:
            origin = self.agent
        if len(origin) != Ndim:
            print(origin)
            raise Exception("The origin coordinates are not consistent with the number of dimensions")

        coord = np.mgrid[tuple([range(N)] * Ndim)]
        for i in range(Ndim):
            coord[i] -= origin[i]
        d = np.zeros([N] * Ndim)
        if norm == 'Manhattan':
            for i in range(Ndim):
                d += np.abs(coord[i])
            return d
        elif norm == 'Euclidean':
            for i in range(Ndim):
                d += (coord[i]) ** 2
            d = np.sqrt(d)
            return d
        elif norm == 'Chebyshev':
            d = np.amax(np.abs(coord), axis=0)
            return d
        else:
            raise Exception("This norm is not implemented")

    def _volume_ball(self, r, Ndim=None, norm=None):
        if Ndim is None:
            Ndim = self.Ndim
        if norm is None:
            norm = self.norm_Poisson
        if norm == 'Manhattan':
            pm1 = 1
        elif norm == 'Euclidean':
            pm1 = 1 / 2
        elif norm == 'Chebyshev':
            pm1 = 0
        else:
            raise Exception("This norm is not implemented")
        return (2 * Gamma(pm1 + 1) * r) ** Ndim / Gamma(Ndim * pm1 + 1)

    def _is_agent_near_boundaries(self, n_boundary):
        # is the agent within n_boundary cell(s) of a boundary of the computational domain?
        for axis in range(self.Ndim):
            if (self.agent[axis] >= self.N - 1 - n_boundary) or (
                    self.agent[axis] <= n_boundary
            ):
                return 1
        return 0

    def _is_agent_stuck(self):
        agent_stuck = False
        if self._agentoo == self.agent:
            self._repeated_visits += 1
        else:
            self._repeated_visits = 0
        if self._repeated_visits > 8:
            agent_stuck = True
        return agent_stuck

    def _set_mu0_Poisson(self, ):
        """Sets the value of mu0_Poisson (mean number of hits at distance = lambda), which is derived from the
         physical dimensionless parameters of the problem. It is required by _mean_number_of_hits().
        """
        dx_over_a = 2.0  # agent step size / agent radius
        lambda_over_a = self.lambda_over_dx * dx_over_a
        a_over_lambda = 1.0 / lambda_over_a

        if self.Ndim == 1:
            mu0_Poisson = 1 / (1 - a_over_lambda) * np.exp(-1)
        elif self.Ndim == 2:
            mu0_Poisson = 1 / np.log(lambda_over_a) * kn(0, 1)
        elif self.Ndim == 3:
            mu0_Poisson = a_over_lambda * np.exp(-1)
        elif self.Ndim > 3:
            mu0_Poisson = (self.Ndim - 2) / Gamma(self.Ndim / 2) / (2 ** (self.Ndim / 2 - 1)) * \
                          a_over_lambda ** (self.Ndim - 2) * kv(self.Ndim / 2 - 1, 1)
        else:
            raise Exception("problem with Ndim")

        mu0_Poisson *= self.R_dt
        self.mu0_Poisson = mu0_Poisson

    def _extract_N_from_2N(self, input, origin):
        if len(origin) != self.Ndim:
            raise Exception("_extract_N_from_2N: len(origin) is different from Ndim")
        if input.shape[-1] == 2 * self.N + 1:
            index = np.array([self.N] * self.Ndim) - origin
        elif input.shape[-1] == 2 * self.N - 1:
            index = np.array([self.N - 1] * self.Ndim) - origin
        else:
            raise Exception("_extract_N_from_2N(): dimension of input must be 2N-1 or 2N+1")
        if self.Ndim == 1:
            output = input[..., index[0]:index[0] + self.N]
        elif self.Ndim == 2:
            output = input[...,
                         index[0]:index[0] + self.N,
                         index[1]:index[1] + self.N]
        elif self.Ndim == 3:
            output = input[...,
                         index[0]:index[0] + self.N,
                         index[1]:index[1] + self.N,
                         index[2]:index[2] + self.N]
        elif self.Ndim == 4:
            output = input[...,
                         index[0]:index[0] + self.N,
                         index[1]:index[1] + self.N,
                         index[2]:index[2] + self.N,
                         index[3]:index[3] + self.N]
        else:
            raise Exception("_extract_N_from_2N() not implemented for Ndim > 4")
        return output

    # __ INPUT TO VALUE FUNCTION _______________________________________
    
    def _inputshape(self, ):
        shape = tuple([2 * self.N - 1] * self.Ndim)
        return shape

    def _centeragent(self, p, agent):
        """Return the probability density of the source centered on the agent

        Args:
            p (numpy array): initial probability in a non-centered environment
            agent (list): vector position of the agent

        Returns:
            numpy array: probability density centered on the agent (tensor of size (2 * N - 1) ** Ndim)
        """
        result = np.zeros([2 * self.N - 1] * self.Ndim)
        if self.Ndim == 1:
            result[self.N - 1 - agent[0]:2 * self.N - 1 - agent[0]] = p
        elif self.Ndim == 2:
            result[
                self.N - 1 - agent[0]:2 * self.N - 1 - agent[0],
                self.N - 1 - agent[1]:2 * self.N - 1 - agent[1],
            ] = p
        elif self.Ndim == 3:
            result[
                self.N - 1 - agent[0]:2 * self.N - 1 - agent[0],
                self.N - 1 - agent[1]:2 * self.N - 1 - agent[1],
                self.N - 1 - agent[2]:2 * self.N - 1 - agent[2],
            ] = p
        else:
            raise Exception("centeragent: dimension not allowed")

        return result
