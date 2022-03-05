"""Provides an OpenAI Gym interface for the source-tracking POMDP."""
import gym
import numpy as np


class GymWrapper(gym.Env):
    """OpenAI Gym interface for the source-tracking POMDP.

    Args:
        sim (SourceTracking): an instance of the SourceTracking class with draw_source = True
        stop_t (int, optional): maximum number of timesteps, it should be large enough to (almost) never be reached
    """

    def __init__(
            self,
            sim,
            stop_t=100000,  # should be large enough to (almost) never be reached
    ):
        super().__init__()

        self.sim = sim      # instance of SourceTrackingBounce
        if not self.sim.draw_source:
            raise Exception("GymWrapper requires draw_source = True")

        self.action_space = gym.spaces.Discrete(self.sim.Nactions)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=tuple([2 * self.sim.N - 1] * self.sim.Ndim), dtype=np.float32)

        self.stop_t = stop_t

    def step(self, action):
        """Make a step in the environment.

        Args:
            action (int): the action to execute

        Returns:
            observation (ndarray): p_source centered on the agent
            reward (float): always a -1 time penalty
            done (bool): whether the search is over
            info (dict): contains the hits received, whether the maximum number of steps is reached, and whether the source was found
        """

        if self.beyond_done:
            raise Exception("Cannot call step() once done, need to reset first!")

        # make a step in source-tracking sim
        hit, _, found = self.sim.step(action, quiet=True)

        # increment steps
        self.t += 1

        # reward = time penalty
        reward = -1

        # timeout
        timeout = False
        if self.t >= self.stop_t:
            timeout = True

        # observation
        observation = np.array(self.sim._centeragent(self.sim.p_source, self.sim.agent), dtype=np.float32)

        # info
        info = {
            "hit": hit,
            "timeout": timeout,
            "found": found,
        }

        # done
        done = False
        if found or timeout:
            done = True
            self.beyond_done = True

        return observation, reward, done, info

    def reset(self, ):
        """Reset the search.

        Returns:
            observation (ndarray): p_source centered on the agent
        """
        self.t = 0
        self.beyond_done = False
        self.sim.restart()
        observation = np.array(self.sim._centeragent(self.sim.p_source, self.sim.agent), dtype=np.float32)
        return observation


