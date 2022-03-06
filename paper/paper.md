---
title: 'OTTO: A Python package to simulate, solve and visualize the source-tracking POMDP'
tags:
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
Solutions to this problem, even approximate, have direct applications to sniffer robots used to track chemicals emitted by explosives, drugs or chemical leaks [@Russell1999;@Marques2006]. 
They may also shed light on how cognitive animals use olfaction to search for food and mates [@Vickers2000;@Reddy2022].

In the source-tracking POMDP, the agent must find a source of odor hidden in a grid world. 
At each step, the agent chooses a neighbor cell where to move next.
Once in a new cell, the agent receives an observation (odor detection) that provides some partial information on how far the source is. 
The search terminates when the agent enters the cell containing the source.
Solving the POMDP means finding the optimal way of choosing moves (policy) so as to reach the source in the smallest possible number of steps.

Computating the optimal policy is not possible, and the challenge resides in finding good approximate policies. A strong baseline is provided by "infotaxis", a heuristic policy devised by @Vergassola2007. 
It has become popular in robotics [@Moraud2010;@Lochmatter2010thesis] and to interpret animal searches [@Vergassola2007;@Voges2014;@Calhoun2014]. 

Several variants have been proposed since [@Masson2013;@Ristic2016;@Karpas2017;@Hutchinson2018;@Chen2020], but the quest for better policies has been hindered by the lack of a trustable, open-source implementation of the source-tracking POMDP.
Existing comparisons of policies are not reliable because no well-defined methodology for policy evaluation exists. 
Besides, the recent successes of reinforcement learning for solving complex navigation problems in turbulence [@Reddy2016;@Alageshan2020] calls for an adaptation of these methods to the source-tracking POMDP.

The target audience of OTTO consists of researchers in biophysics, applied mathematics and robotics working on optimal strategies for olfactory searches in turbulent conditions.

# Summary

`OTTO` (short for Odor-based Target Tracking Optimization) is a Python 
package to visualize, evaluate and learn strategies for odor-based searches.

`OTTO` provides:

  1. a simulator of the source-tracking POMDP for any number of space dimensions, domain sizes and source intensities, together with rendering in 1D, 2D and 3D;
  2. an implementation of several heuristic policies including "infotaxis" [@Vergassola2007] and its recently proposed "space-aware" variant [@Loisy2022];
  3. a parallelized algorithm to evaluate policies (probability of finding the source, distribution of search times, etc.) using a rigorous, well-defined protocol;
  4. a custom model-based deep reinforcement learning algorithm for training neural-network policies, together with a library ("zoo") of trained neural networks that achieve near-optimal performance;
  5. a wrapper of the source-tracking POMDP that follows the OpenAI Gym interface.

`OTTO` aims at facilitating future research:

  1. New heuristic policies can easily be implemented, visualized, and evaluated. To facilitate comparison to existing baselines, the performance of several policies (including infotaxis and near-optimal) is reported in a freely available dataset generated with `OTTO` [@dataset].
  2. The gym wrapper makes the source-tracking POMDP easily accessible to the reinforcement learning community. OpenAI Gym [@gym] is the _de facto_ standard for simulators. It is compatible with most general-purpose model-free reinforcement learning libraries (e.g., Stable Baselines [@stable-baselines3], OpenAI-Baselines [@openai-baselines], RLlib [@RLlib], CleanRL [@CleanRL], ChainerRL/PFRL [@PFRL]).

# Mentions

The methodological aspects of `OTTO` (generalization of the POMDP to an arbitrary number of space dimensions, policy evaluation protocol, model-based reinforcement learning algorithm) have been developed as part of a publication by its authors [@Loisy2022] (currently under review, preprint on arxiv).

# Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 834238).

# References

