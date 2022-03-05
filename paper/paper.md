---
title: 'OTTO: A Python package to simulate, solve and visualize the source-tracking POMDP'
tags:
  - Python
  - POMDP
  - olfactory search
  - reinforcement learning
authors:
  - name: Aurore Loisy^[corresponding author]
    orcid: 0000-0002-8089-8636
    affiliation: 1
  - name: Christophe Eloy^[corresponding author]
    orcid: 0000-0003-4114-7263
    affiliation: 1
affiliations:
 - name: Aix Marseille Univ, CNRS, Centrale Marseille, IRPHE, Marseille, France
   index: 1
date: 7 March 2022
bibliography: paper.bib

---

# Statement of need

The source-tracking problem is a POMDP (partially observable Markov decision process) designed by @Vergassola2007 to mimic the problem of searching for a source of odor in a turbulent flow. 
Far from a "toy" POMDP, it incorporates physical models of odor dispersion and detection that reproduce the major features of olfactory searches in turbulence. 
Solutions to this problem, even approximate, have direct applications to sniffer robots used to track chemicals emitted by explosives, drugs or chemical leaks [@Russell1999;@Marques2006;@Moraud2010]. 
They may also shed light on how cognitive animals use olfaction to search for food and mates [@Vickers2000;@Vergassola2007;@Voges2014;@Calhoun2014].

In the source-tracking POMDP, the agent must find a source of odor hidden in a grid world. 
At each step, the agent chooses a neighbor cell where to move next.
Once in a new cell, the agent receives an observation (odor detection), which provides some partial information on how far the source is. 
The search terminates when the agent enters the cell containing the source.
Solving the POMDP means finding the optimal way of choosing moves (policy) so as to reach the source in the smallest possible number of steps.

Computating the optimal policy is not possible, and the challenge resides in finding good approximations. A strong baseline is provided by "infotaxis", a heuristic policy proposed in 2007 by @Vergassola2007. 
Since then, a number of variants and alternatives have been proposed [@Masson2013;@Ristic2016;@Karpas2017;@Hutchinson2018;@Chen2020]

But the quest for better policies has been hindered by the lack of a trustable, open-source implementation of the source-tracking POMDP.
Comparing policies across publications is unreliable because no well-defined methodology for policy evaluation exists. 
Besides, the recent successes of reinforcement learning for solving complex navigation problems in turbulence [@Reddy2016;@Alageshan2020] calls for an adaptation of these methods to the source-tracking POMDP.


# Summary

`OTTO` (short for Odor-based Target Tracking Optimization) is a Python 
package to visualize, evaluate and learn strategies for odor-based searches.

`OTTO` provides:
  - a simulator of the source-tracking POMDP for any number of space dimensions, domain sizes and source intensities, together with rendering in 1D, 2D and 3D;
  - an efficient algorithm to evaluate policies using a rigorous, well-defined protocol;
  - several heuristic policies including "infotaxis" [@Vergassola2007] and its recently proposed "space-aware" variant [@Loisy2022];
  - a custom deep reinforcement learning algorithm for training neural-network policies, together with a library ("zoo") of trained neural networks that achieve near-optimal performance;
  - a wrapper of the source-tracking POMDP that follows the OpenAI Gym interface.

`OTTO` facilitates future research:
  - New heuristic policies can easily be implemented, visualized, and evaluated. Their performance can be compared to that of other policies, and to the near-optimal performance for a number of cases where this performance is known.
  - The gym wrapper makes the source-tracking POMDP easily accessible to the reinforcement learning community. OpenAI Gym [@gym] is a _de facto_ standard for environment simulators. It is compatible with most general purpose reinforcement learning libraries (e.g., Stable Baselines [@stable-baselines3], OpenAI-Baselines [@openai-baselines], RLlib [@RLlib], CleanRL [@CleanRL], ChainerRL/PFRL [PFRL]).


# Mentions

`OTTO` has been used in a publication by its authors [@Loisy2022] (currently under review, preprint on arxiv).

# Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 834238).

# References

