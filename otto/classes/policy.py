#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic policy class and functions"""


def policy_name(policy_index):
    """Returns the policy name associated to policy_index"""
    if policy_index == -1:
        name = "RL"
    elif policy_index == 0:
        name = "infotaxis"
    elif policy_index == 1:
        name = "space-aware infotaxis"
    elif policy_index == 2:
        name = "custom policy"
    elif policy_index == 5:
        name = "random walk"
    elif policy_index == 6:
        name = "greedy"
    elif policy_index == 7:
        name = "mean distance"
    elif policy_index == 8:
        name = "voting"
    elif policy_index == 9:
        name = "most likely state"
    else:
        raise Exception("The policy " + str(policy_index) + " does not exist (yet)!")
    return name


class Policy:
    """
        A generic policy template.

        Args:
            policy (int):
                    - -1: reinforcement learning
                    - 0: infotaxis (Vergassola, Villermaux and Shraiman, Nature 2007)
                    - 1: space-aware infotaxis
                    - 2: custom policy (to be implemented by the user)
                    - 5: random walk
                    - 6: greedy policy
                    - 7: mean distance policy
                    - 8: voting policy (Cassandra, Kaelbling & Kurien, IEEE 1996)
                    - 9: most likely state policy (Cassandra, Kaelbling & Kurien, IEEE 1996)

        Attributes:
            policy_index (int): policy index
            policy_name (str): name of the policy

    """
    def __init__(
            self,
            policy,
    ):
        self.policy_index = policy
        self.policy_name = policy_name(self.policy_index)

    def choose_action(self, ):
        """Choose an action based on the current belief (env.p_source, env.agent), according to the policy.

        Returns:
            action_chosen (int): chosen action
        """
        action_chosen = self._choose_action()

        return action_chosen
