
# OTTO

[![Documentation Status](https://readthedocs.org/projects/otto-c0pep0d/badge/?version=latest)](https://otto-c0pep0d.readthedocs.io/en/latest/?badge=latest)


**OTTO** (short for **O**dor-based **T**arget **T**racking **O**ptimization) is a Python package that provides the means to 
**visualize, evaluate and learn** policies for the source-tracking problem, a POMDP designed to provide a test-bed
for **odor-based search strategies** such as the popular **"infotaxis"**.

OTTO is part of the [C0PEP0D](https://C0PEP0D.github.io/) project and has been used in a 
[publication](https://arxiv.org/abs/2112.10861).

## Table of contents
* [Background](#background)
    * [Motivation](#motivation)
    * [The source-tracking POMDP](#the-source-tracking-pomdp)
    * [What does OTTO do?](#what-does-otto-do)
    * [Infotaxis](#infotaxis)
    * [Space-aware infotaxis](#space-aware-infotaxis)
    * [Reinforcement learning](#reinforcement-learning)
* [Installation](#installation)
    * [Requirements](#requirements)
    * [Conda users](#conda-users)
    * [Installing](#installing)
    * [Testing](#testing)
* [How to use OTTO?](#how-to-use-otto)
    * [First steps](#first-steps)
    * [Changing parameters](#changing-parameters)
    * [Evaluating a policy](#evaluating-a-policy)
    * [Learning a policy](#learning-a-policy)
    * [Visualizing and evaluating a learned policy](#visualizing-and-evaluating-a-learned-policy)
    * [Pre-trained RL policies](#pre-trained-rl-policies)
    * [Custom policies](#custom-policies)
    * [Cleaning up](#cleaning-up)
* [Documentation](#documentation)
* [Known issues](#known-issues)
    * [All users](#all-users)
    * [Windows users](#windows-users)
* [Community guidelines](#community-guidelines)
    * [Reporting bugs](#reporting-bugs)
    * [Contributing](#contributing)
    * [Getting help](#getting-help)
* [Authors](#authors)
* [How to cite OTTO?](#how-to-cite-otto)
* [License](#license)
* [Acknowledgements](#acknowledgements)



## Background

### Motivation
Imagine a treasure hunt where the player needs to find a hidden treasure using odor cues. 
Because the wind constantly changes direction, the player smells nothing most of the time, but occasionally catches 
a puff. How should he move to find the treasure as fast as possible?
This game is a common task, for example, mosquitoes looking for a prey to bite by detecting carbon dioxide or 
sniffer robots trying to locate explosive in an airport. 
Because of turbulence, there is no odor trail to follow in this problem, which makes it particularly challenging.

### The source-tracking POMDP

The source-tracking problem is a POMDP (partially observable Markov decision process) 
where the agent (the searcher) must find, as fast as possible, 
a stationary target (the source) hidden in a grid world 
using stochastic observations (odor detections, called "hits").
It was originally designed by 
Vergassola et al. (Nature, 2007) 
to mimic the task faced by animals or robots searching for a source of odor in a turbulent flow.

Finding the optimal policy (strategy) using exact methods is not possible.
Yet various heuristic policies have been proposed over the years, 
and the problem is also amenable to reinforcement learning.

### What does OTTO do?

OTTO provides an efficient implementation of the source-tracking POMDP (in any number of space dimensions), 
of various heuristic policies (including infotaxis), 
as well as a reinforcement learning algorithm able to learn near-optimal policies.

OTTO also provides an efficient method to evaluate policies (including custom policies defined by the user)
using a rigorous protocol. 

Finally, OTTO allows you to visualize and record videos of searches (only up to 3D!).

### Infotaxis

Infotaxis is a heuristic policy proposed by Vergassola et al. (Nature, 2007). 
It states that the agent should choose the action from which it expects the greatest information gain about 
the source location. 

The physical intuition behind this strategy is, quoting the authors, that 
"information accumulates faster close to the source because cues arrive at a higher rate, 
hence tracking the maximum rate of information acquisition will guide the searcher to the source much like 
concentration gradients in chemotaxis".

Infotaxis is far superior to all naive strategies, such as going to the more likely source location.
Yet it is known to be suboptimal.

### Space-aware infotaxis

Space-aware infotaxis is variant of infotaxis which has been shown to beat infotaxis in most cases.

### Reinforcement learning

Near-optimal policies can be obtained using deep reinforcement learning.
The training algorithm is a model-based version of DQN (Mnih et al., Nature, 2015).



## Installation

### Requirements

OTTO requires Python 3.8 or greater.
Dependencies are listed in [requirements.txt](https://github.com/C0PEP0D/otto/requirements.txt),
missing dependencies will be automatically installed.

Optional: OTTO requires `ffmpeg` to make videos.
If `ffmpeg` is not installed, OTTO will save video frames as images instead.

### Conda users

If you use conda to manage your Python environments, you can install OTTO in a dedicated environment `ottoenv`

``` bash
conda create --name ottoenv python=3.8
conda activate ottoenv
```

### Installing
First download the package or clone the git repository with

``` bash
git clone https://github.com/C0PEP0D/otto.git
```

Then go to the `otto` directory and install OTTO using

``` bash
python3 setup.py install
```

### Testing
You can test your installation with following command:

```bash
pytest tests
```
This will execute the Python pytest functions located in the folder `tests`.

## How to use OTTO?

### First steps

Go to the `otto` subdirectory.

You will see that it is organized in **three main directories** corresponding to the **three main uses of OTTO**:

- `evaluate`: for **evaluating the performance** of a policy
- `learn`: for **learning a neural network policy** that solves the task
- `visualize`: for **visualizing a search** episode

The other directory, `classes`, contains all the class definitions used by the main scripts.

The three main directories share the same structure. They contain:

- `*.py`: the main script
- `parameters`: the directory to store input parameters
- `outputs`: the directory to store files generated by the script (it will be created on first use)

To use OTTO, go to the relevant main directory and run the corresponding script. 
For example, to visualize an episode, go to the `visualize` directory and run the `visualize.py` script with

```bash
python3 visualize.py
```

You should now see the rendering of a 1D search in a new window (it may be very short!).
You can visualize another episode by using again the same command.

Some logging information is displayed in the terminal as the script runs.
In the rendering window, the first panel is a map of hits, and the second panel is the agent's belief 
(probability distribution over source locations).

The videos have been saved as `visualize/outputs/YYmmdd-HHMMSS_video.mp4` where 'YYmmdd-HHMMSS' is a 
timestamp (the time you started the script). 

If you do not have `ffmpeg`, or if you are using Windows, or if there was a problem with video making, you will find instead frames saved
in `visualize/outputs/YYmmdd-HHMMSS_frames`.


### Changing parameters

Many parameters (space dimension, domain size, source intensity, policy, ...) can be changed from the defaults. 
To run a script with different parameters, create a Python script that sets your parameters in the
`parameters` directory.

A file `myparam.py` is already present in `visualize/parameters/` for this example. 
It contains a single line

```python
N_DIMS = 2
```

which sets the dimensionality of the search to 2D (1D is the default).

User-defined parameters are called by using the `--input` option followed by the name of the parameter file. 
For example, you can now visualize a search in 2D with

```bash
python3 visualize.py --input myparam.py
```

The `--input` option can be shortened to `-i`, and the file name can be with or without `.py`. So the command

```bash
python3 visualize.py -i myparam
```
will have the same effect.

Each `parameters` directory contain sample parameter files called `example*.py`. 
They show the parameters you can play with, for example:

- `N_DIMS` sets the dimensionality of the search (1D, 2D, 3D), default is `N_DIMS = 1`
- `LAMBDA_OVER_DX` controls the size the domain, default is `LAMBDA_OVER_DX = 2.0`
- `R_DT` controls the source intensity, default is `R_DT = 2.0`
- `POLICY` defines the policy to use, default is `POLICY = 0` (infotaxis)

Note: for advanced users, you can access the list of *all* parameters and their default values by examining
the contents of `__defaults.py`.

### Evaluating a policy

The `evaluate.py` script (in the `evaluate` directory) computes many statistics that characterize the search, such as

- probability of never finding the source,
- average time to find the source,
- probability distribution of arrival times,
- and much more.

It does so essentially by running thousands of episodes in parallel and averaging over those.

You can try with

```bash
python3 evaluate.py
```

This will take some time (order of magnitude is 2 minutes on 8 cores). 
Logging information is displayed in the terminal while the episodes are running.

Windows users: if a `NameError` is raised, see [known issues](#known-issues).

Once the script has completed, you can look at the results in the directory `evaluate/outputs/YYmmdd-HHMMSS` 
where 'YYmmdd-HHMMSS' is the time you started the script.
`Ymmdd-HHMMSS_figure_distributions.pdf` is a figures summarizing the results. 
All output files are described in the [documentation](#documentation).

These results are for the "infotaxis" policy, which is the default policy.
You can now try to compute the statistics of another policy on the same problem. 
For example, evaluate the "space-aware infotaxis" policy by running

```bash
python3 evaluate.py --input myparam.py
```

where `myparam.py` is a file containing the line 

```python
POLICY = 1
```

This file is already present in `evaluate/parameters/` for this example. 

The main policies are

- `POLICY = 0` for infotaxis (default)
- `POLICY = 1` for space-aware infotaxis
- `POLICY = -1` for a reinforcement learning policy: for that we need to learn a model first!

All policies are described in the [documentation](#documentation).


### Learning a policy

The `learn.py` script learns a policy using reinforcement learning (RL). 
It actually trains a neural network model of the optimal value function.
The (approximately) optimal policy is then derived from this function.

To train a model, go to the `learn` directory and use

```bash
python3 learn.py
```

Now is the perfect time for a coffee since it will take quite a while. Logging information is displayed in the terminal while the 
script runs (if the script seems to have frozen, see [known issues](#known-issues)).

When you come back, you can look at the contents of the `learn/outputs/YYmmdd-HHMMSS` directory.
There should be a figure called `YYmmdd-HHMMSS_figure_learning_progress.png` (if not you need a larger coffee).

This figure shows the progress of the learning agent and is periodically updated as the training progresses. 
In particular, it shows the evolution of 'p_not_found', the probability that the source is never found, and of 'mean', 
the mean time to find the source provided it is ever found. 
Note that the mean is meaningless if p_not_found is larger than 1e-3, in that case it is depicted by a cross.

Other outputs are explained in the [documentation](#documentation).

Completing the training may take up to roughly 5000-10000 iterations (several hours on an 
average laptop), but progress should be clearly visible from 500-1000 iterations. 
For reference, the trained network should achieve p_not_found < 1e-6 and mean ~ 7.1-7.2, which is essentially optimal.

Training will continue until 10000 iterations, but can be stopped at any time.

Models are saved in the `learn/models/YYmmdd-HHMMSS` directory:

- `YYmmdd-HHMMSS_model` is the most recent model,
- `YYmmdd-HHMMSS_model_bkp_i`, where i is an integer, are the models saved at evaluation points
  (the models which performance is shown in `YYmmdd-HHMMSS_figure_learning_progress.png`).

Note: training can restart from a previously saved model.

### Visualizing and evaluating a learned policy

Once a model is trained, the learned policy can be evaluated or visualized using the corresponding scripts.
For that, simply run the scripts with a parameter file (using `--input`) containing

```python
POLICY = -1
MODEL_PATH = "../learn/models/YYmmdd-HHMMSS/YYmmdd-HHMMSS_value_model_bkp_i"
```

where `MODEL_PATH` is the path to the model you want to evaluate or visualize.

Important: parameters should be consistent. For example, if you set `N_DIMS = 2` for learning then you must also 
set `N_DIMS = 2` for evaluation and visualization.

### Pre-trained RL policies

***TODO: this part is in construction (current pre-trained model is just for testing that the script runs, 
actual pre-trained models will be added soon)***

A collection of pre-trained models are provided in the `zoo` directory accessible from the root of the package. 
Models are saved in the `models` directory and corresponding parameter files are in the `parameters` directory.
Pre-trained models are named `zoo_model_i` where i is an integer.

To visualize the RL policy given by `zoo_model_1`, use

```bash
python3 visualize.py --input zoo_model_1
```

Similarly you can evaluate this model with

```bash
python3 evaluate.py --input zoo_model_1
```

Note that since `zoo_model_1.py` is not present in the `visualize/parameters` directory, the script will automatically
search for it in the `zoo/parameters` directory.

### Custom policies

You want to try your own policy? 
Policies are implemented in `classes/heuristicpolicies`. 
You can define your own in the function `_custom_policy`. 

To use it in the main scripts, set `POLICY = 2` in your parameter file.


### Cleaning up

The directories can be restored to their original state by running the `cleanall.sh` bash script located 
at the root of the package.
Warning: all user-generated outputs and models will be deleted!


## Documentation
***TODO: documentation is currently in construction***

**OTTO** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for documentation and is made available online 
[here](https://otto-c0pep0d.readthedocs.io/en/latest/). 
To build the html version of the docs locally, go to the `docs` directory and use:

```bash
make html
```

The generated html can be viewed by opening `docs/_build/html/index.html`.

## Known issues

### All users

When using large neural networks in parallel, the code may hang. This is a 
[known incompatibility](https://github.com/keras-team/keras/issues/9964)
between `keras` and `multiprocessing`.
The workaround is to set `N_PARALLEL = 1` in the parameter file, which enforces sequential computations.

### Windows users

While OTTO can run on all platforms, it has been developed for Unix-based systems
and there are minor issues with Windows.

1. Videos are not recorded by `visualize.py`. Frames are saved as images instead.
2. Parallelization for `learn.py` and `evaluate.py` does not currently work, 
and the error `NameError: name '*' is not defined` is raised when running these scripts.
This is because child processes instanciated with `multiprocessing` do not see global variables defined only during execution (after `if __name__ == "__main__"`).
The workaround is to set `N_PARALLEL = 1` in the parameter file, which enforces sequential computations 



## Community guidelines

### Reporting bugs
If you discover a bug in OTTO which is not a [known issue](#known-issues), please create a 
[new issue](https://github.com/C0PEP0D/otto/issues/new).

### Contributing
Have you designed a new policy? Would you like to add a new feature? Can you fix a [known issue](#known-issues)?
We welcome contributions to OTTO.
To contribute, please fork the repository and submit a pull request.

### Getting help
Are you having troubles with OTTO? Please first consult the instructions for [installing](#installation) 
and [using](#how-to-use-otto) OTTO, check the [known issues](#known-issues), and explore the [documentation](#documentation).

Can you still not find an answer? Would you like more information?
Please create [an issue](https://github.com/C0PEP0D/otto/issues/new) or send an email with the 
subject "OTTO: your request" to the [authors](#authors).

## Authors

**OTTO** is developed by 
[Aurore Loisy](mailto:aurore.loisy@gmail.com) and 
[Christophe Eloy](mailto:christophe.eloy@centrale-marseille.fr)
*(Aix Marseille Univ, CNRS, Centrale Marseille, IRPHE, Marseille, France)*.

## How to cite OTTO?

If you use **OTTO** in your publications, you can cite the package as follows:

> OTTO: a Python package to simulate, solve and visualize the source-tracking POMDP. http://github.com/C0PEP0D/otto

or if you use LaTeX:

```tex
@misc{OTTO,
  author = {Loisy, A. and Eloy, C.},
  title = {{OTTO}: a {P}ython package to simulate, solve and visualize the source-tracking {POMDP}},
  howpublished = {\url{http://github.com/C0PEP0D/otto}}
}
```

## License

See the [LICENSE](LICENSE) file for license rights and limitations.

## Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 
research and innovation programme (grant agreement No 834238).

