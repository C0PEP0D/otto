#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script ``learn.py`` is used to learn the optimal value function using a neural network model.

The training algorithm is a value-based version of the DQN algorithm (Mnih et al., Nature 2015).

The script periodically evaluates the performance of the RL policy derived from the current value function.

Training can be interrupted at any time, and be restarted from a previously saved model.

Outputs generated during training are:
    - \*figure_learning_progress.png
        It monitors the progress of the learning agent and is periodically updated as the training progresses.
        It shows the evolution of

            - 'eps': current value of eps (probability of taking a random action) used when collecting new experience
            - 'p_not_found': probability that the source is never found
            - 'mean': mean number of steps to find the source, provided it is ever found (if 'p_not_found' > 1e-3,
              the mean is depicted by a cross instead of a dot)
            - 'rel_mean': 'mean' divided by the mean obtained for the reference policy
            - 'p50': number of steps to find the source with 50 % probability
            - 'p99': number of steps to find the source with 99 % probability

        as a function of training iterations (top row) and number of transitions used for training (bottom row).
    - \*figure_stats_i.png
        A figure is produced every time the learning agent is evaluated (i is an integer).
        It shows distributions (pdf, cdf, ccdf) and statistics (mean, standard deviation, median, etc.)
        of the number of steps to find the source.
    - \*table_stats.npy
        Numpy array containing the history of performance.
        Each row corresponds to a training iteration where the performance was evaluated, and the columns contain

            - col 0: training iteration
            - col 1: number of transitions seen
            - col 2: number of transitions generated
            - col 3: eps (probability of taking a random action)
            - col 4: probability that the source is never found
            - col 5: mean number of steps to find the source, provided that the source is ultimately found
            - col 6: error on the mean
            - col 7: number of steps to find the source with 50 % probability
            - col 8: number of steps to find the source with 99 % probability
    - \*parameters.txt
        Text file summarizing the parameters used.

Models are saved periodically:
    - \*model
        most recent model
    - \*model_bkp_i
        models saved at evaluation points (i is an integer), that is, models which performance is shown in figures

Parameters of the script are:
    - Source-tracking POMDP
        - N_DIMS (int > 0)
            number of space dimensions (1D, 2D, ...)

            Warning: while the script works for any space dimension, it is very computationally intensive and is generally not
            usable in more than 2D.
        - LAMBDA_OVER_DX (float >= 1.0)
            dimensionless problem size
        - R_DT (float > 0.0)
            dimensionless source intensity
        - NORM_POISSON ('Euclidean', 'Manhattan' or 'Chebyshev')
            norm used for hit detections, usually 'Euclidean'
        - N_HITS (int >= 2 or None)
            number of possible hit values, set automatically if None
        - N_GRID (int >= 3 or None)
            linear size of the domain, set automatically if None

    - Reinforcement learning
        Neural network architecture (fully connected)
            - FC_LAYERS (int > 0)
                number of hidden layers
            - FC_UNITS (int > 0 or tuple(int > 0))
                number of units per layers

        Stochastic gradient descent
            - BATCH_SIZE (int > 0)
                size of the mini-batch
            - N_GD_STEPS (int > 0)
                number of gradient descent steps per training iteration
            - LEARNING_RATE (0.0 < float < 1.0)
                usual learning rate

        Exploration: eps is the probability of taking a random action
            - E_GREEDY_FLOOR (0.0 <= float <= 1.0)
                floor value of eps
            - E_GREEDY_0 (0.0 <= float <= 1.0)
                initial value of eps
            - E_GREEDY_DECAY (float > 0.0)
                timescale for eps decay, in number of training iterations

        Accounting for symmetries:
            - SYM_EVAL_ENSEMBLE_AVG (bool)
                whether to average value over symmetric duplicates during evaluation
            - SYM_TRAIN_ADD_DUPLICATES (bool)
                whether to augment data by including symmetric duplicates with identical targets during training step
            - SYM_TRAIN_RANDOMIZE (bool)
                whether to apply random symmetry transformations when generating the data (no duplicates)

        Experience replay
            - MEMORY_SIZE (int > 0)
                number of transitions (s, s') to keep in memory
            - REPLAY_NTIMES (int > 0)
                how many times a transition is used for training before being deleted, on average

        Additional DQN algo parameters
            - ALGO_MAX_IT (int > 0)
                max number of training iterations
            - UPDATE_FROZEN_MODEL_EVERY (int > 0)
                how often is the target network updated, in number of training iterations
            - DDQN (bool)
                whether to use Double DQN instead of original DQN

        Evaluation of the RL policy
            - POLICY_REF (int)
                heuristic policy to use for comparison
            - EVALUATE_PERFORMANCE_EVERY (int > 0)
                how often is the RL policy evaluated, in number of training iterations
            - N_RUNS_STATS (int > 0 or None)
                number of episodes used to compute the stats of a policy, set automatically if None

        Monitoring/Saving during the training
            - PRINT_INFO_EVERY (int > 0)
                how often to print info on screen, in number of training iterations
            - SAVE_MODEL_EVERY (int > 0)
                how often to save the current model, in number of training iterations
                (in addition, model copies will be saved every EVALUATE_PERFORMANCE_EVERY)

    - Criteria for episode termination
        - STOP_t (int > 0 or None)
            maximum number of steps per episode, set automatically if None
        - STOP_p (float ~ 0.0)
            episode stops when the probability that the source has been found is greater than 1 - STOP_p

    - Parallelization
        - N_PARALLEL (int)
            number of episodes computed in parallel when generating new experience or evaluating the RL policy
            (if <= 0, will use all available cpus)

            Known bug: for large neural networks, the code may hang if N_PARALLEL > 1, so use N_PARALLEL = 1 instead.

    - Reload an existing model
        - MODEL_PATH (str or None)
            path of the model (neural network) to reload, if None starts from scratch

    - Saving
        - RUN_NAME (str or None)
            prefix used for all output files, if None will use a timestamp
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], '..', '..')))
import time
import argparse
import importlib
import random
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy
from itertools import repeat
from otto.classes.training import TrainingEnv as env
from otto.classes.valuemodel import ValueModel, reload_model
from otto.classes.heuristicpolicy import HeuristicPolicy
from otto.classes.policy import policy_name

# import default globals
from otto.learn.parameters.__defaults import *

# import globals from user defined parameter file
if os.path.basename(sys.argv[0]) == "learn.py":
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-i', '--input',
                        dest='inputfile',
                        help='name of the file containing the parameters')
    args = vars(parser.parse_args())
    if args['inputfile'] is not None:
        filename, fileextension = os.path.splitext(args['inputfile'])
        params = importlib.import_module(name="parameters." + filename)
        names = [x for x in params.__dict__ if not x.startswith("_")]
        globals().update({k: getattr(params, k) for k in names})
        del params, names
    del parser, args

# other globals
EPSILON = 1E-10

if N_DIMS > 2:
    if SYM_EVAL_ENSEMBLE_AVG or SYM_TRAIN_ADD_DUPLICATES or SYM_TRAIN_RANDOMIZE:
        raise Exception("Symmetries are not implemented in more than 2 dimensions, "
                        "set all SYM_* parameters to False.")

if N_PARALLEL <= 0:
    N_PARALLEL = os.cpu_count()

CONTINUE_TRAINING = False
if MODEL_PATH is not None:
    CONTINUE_TRAINING = True
    if not os.path.isdir(MODEL_PATH):
        print(MODEL_PATH)
        raise Exception("This model cannot be found")

if RUN_NAME is None:
    RUN_NAME = time.strftime("%Y%m%d-%H%M%S")

DIR_OUTPUTS = os.path.abspath(os.path.join(sys.path[0], "outputs", RUN_NAME))
DIR_MODELS = os.path.abspath(os.path.join(sys.path[0], "models", RUN_NAME))

# config
np.set_printoptions(precision=4)


# Build and compile model ________________________________________________________________
def build_new_model(
        Ndim,
        FC_layers,
        FC_units,
        learning_rate,
):
    """
    Creates and compiles the model.

    Args:
        Ndim (int): number of space dimensions for the search problem
        FC_layers (int): number of hidden layers
        FC_units (int or tuple): units per layer
        learning_rate: usual learning rate

    Returns:
        model (ValueModel): instance of the neural network model
    """

    # Instantiate a new model
    model = ValueModel(
        Ndim=Ndim,
        FC_layers=FC_layers,
        FC_units=FC_units,
    )

    # Compile and build the model
    model.build_graph(input_shape_nobatch=MYENV.NN_input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)

    return model


# Parameters printing utils ________________________________________________________________
def save_parameters(env, model):
    """Save parameters in a file and print them on screen."""
    param_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_parameters" + ".txt"))
    pfile = open(param_file, "a")
    for out in (None, pfile):
        print("Tensorflow version = " + str(tf.__version__), file=out)
        
        print("* Problem parameters", file=out)
        print("N_DIMS = " + str(env.Ndim), file=out)
        print("LAMBDA_OVER_DX = " + str(env.lambda_over_dx), file=out)
        print("R_DT = " + str(env.R_dt), file=out)
        print("MU0_POISSON = " + str(env.mu0_Poisson), file=out)
        print("NORM_POISSON = " + env.norm_Poisson, file=out)
        print("N_GRID = " + str(env.N), file=out)
        print("N_HITS = " + str(env.Nhits), file=out)
        print("STOP_t = " + str(STOP_t), file=out)
        print("STOP_p = " + str(STOP_p), file=out)
        print("EPSILON = " + str(EPSILON), file=out)

        print("* Parallelization", file=out)
        print("N_PARALLEL = " + str(N_PARALLEL), file=out)

        print("* Symmetries", file=out)
        print("SYM_EVAL_ENSEMBLE_AVG = " + str(SYM_EVAL_ENSEMBLE_AVG), file=out)
        print("SYM_TRAIN_ADD_DUPLICATES = " + str(SYM_TRAIN_ADD_DUPLICATES), file=out)
        print("SYM_TRAIN_RANDOMIZE = " + str(SYM_TRAIN_RANDOMIZE), file=out)

        print("* Exploration", file=out)
        print("E_GREEDY_FLOOR = " + str(E_GREEDY_FLOOR), file=out)
        print("E_GREEDY_0 = " + str(E_GREEDY_0), file=out)
        print("E_GREEDY_DECAY = " + str(E_GREEDY_DECAY), file=out)

        print("* Gradient descent", file=out)
        print("BATCH_SIZE = " + str(BATCH_SIZE), file=out)
        print("N_GD_STEPS = " + str(N_GD_STEPS), file=out)
        print("LEARNING_RATE = " + str(LEARNING_RATE), file=out)

        print("* Experience replay", file=out)
        print("MEMORY_SIZE = " + str(MEMORY_SIZE), file=out)
        print("NEW_TRANS_PER_IT = " + str(NEW_TRANS_PER_IT), file=out)
        print("REPLAY_NTIMES = " + str(REPLAY_NTIMES), file=out)

        print("* Other DQN parameters", file=out)
        print("ALGO_MAX_IT = " + str(ALGO_MAX_IT), file=out)
        print("UPDATE_FROZEN_MODEL_EVERY = " + str(UPDATE_FROZEN_MODEL_EVERY), file=out)
        print("DDQN = " + str(DDQN), file=out)

        print("* NN architecture", file=out)
        print("FC_LAYERS = " + str(FC_LAYERS), file=out)
        print("FC_UNITS = " + str(FC_UNITS), file=out)
        Nweights = np.sum([np.prod(w.shape) for w in model.get_weights()])
        print("Number of weights = ", Nweights, file=out)
        print("input shape = ", env.NN_input_shape, file=out)
        if out is None:
            model.summary()
        else:
            model.summary(print_fn=lambda x: out.write(x + '\n'))

        print("* Performance evaluation", file=out)
        print("EVALUATE_PERFORMANCE_EVERY = " + str(EVALUATE_PERFORMANCE_EVERY), file=out)
        print("N_RUNS_STATS = " + str(N_RUNS_STATS), file=out)

        print("* Save parameters", file=out)
        print("SAVE_MODEL_EVERY = " + str(SAVE_MODEL_EVERY), file=out)
        print("SAVE_MODEL_BACKUP_EVERY = " + str(EVALUATE_PERFORMANCE_EVERY), file=out)

        if CONTINUE_TRAINING:
            print("* Reloading parameters", file=out)
            print("MODEL_PATH = " + str(MODEL_PATH), file=out)

    pfile.close()
    sys.stdout.flush()


def param2subtitle(env, model):
    """Create of string summarizing all parameters (useful as a subtitle in plots)."""
    Nweights = np.sum([np.prod(w.shape) for w in model.get_weights()])

    arch = ("FC_LAYERS="
            + str(FC_LAYERS)
            + "    "
            + "FC_UNITS="
            + str(FC_UNITS)
            + "    "
            + "INPUT_SHAPE="
            + str(env.NN_input_shape)
            )

    reloaded_model = None
    if CONTINUE_TRAINING:
        reloaded_model = os.path.basename(MODEL_PATH)

    subtitle = (
            "N_DIMS="
            + str(env.Ndim)
            + "    "
            + "LAMBDA_OVER_DX="
            + str(env.lambda_over_dx)
            + "    "
            + "R_DT = "
            + str(env.R_dt)
            + "    "
            + "NORM_POISSON="
            + str(env.norm_Poisson)
            + "    "
            + "N_GRID="
            + str(env.N)
            + "    "
            + "N_HITS="
            + str(env.Nhits)
            + "    "
            + "STOP_t="
            + str(STOP_t)
            + "    "
            + "STOP_p="
            + str(STOP_p)
            + "\n\n"
            + "SYM_EVAL_ENSEMBLE_AVG="
            + str(SYM_EVAL_ENSEMBLE_AVG)
            + "    "
            + "SYM_TRAIN_ADD_DUPLICATES="
            + str(SYM_TRAIN_ADD_DUPLICATES)
            + "    "
            + "SYM_TRAIN_RANDOMIZE="
            + str(SYM_TRAIN_RANDOMIZE)
            + "    "
            + "N_WEIGHTS="
            + str(Nweights)
            + "    "
            + "RELOADED_MODEL="
            + str(reloaded_model)
            + "    "
            + "\n\n"
            + arch
            + "\n\n"
            + "BATCH_SIZE="
            + str(BATCH_SIZE)
            + "    "
            + "N_GD_STEPS="
            + str(N_GD_STEPS)
            + "    "
            + "LEARNING_RATE="
            + str(LEARNING_RATE)
            + "    "
            + "E_GREEDY_FLOOR="
            + str(E_GREEDY_FLOOR)
            + "    "
            + "E_GREEDY_0="
            + str(E_GREEDY_0)
            + "    "
            + "E_GREEDY_DECAY="
            + str(E_GREEDY_DECAY)
            + "\n\n"
            + "MEMORY SIZE="
            + str(MEMORY_SIZE)
            + "    "
            + "NEW_TRANS_PER_IT="
            + str(NEW_TRANS_PER_IT)
            + "    "
            + "REPLAY_NTIMES="
            + str(REPLAY_NTIMES)
            + "    "
            + "ALGO_MAX_IT="
            + str(ALGO_MAX_IT)
            + "    "
            + "UPDATE_FROZEN_MODEL_EVERY="
            + str(UPDATE_FROZEN_MODEL_EVERY)
            + "    "
            + "DDQN="
            + str(DDQN)
            + "    "
            + "EVALUATE_PERFORMANCE_EVERY="
            + str(EVALUATE_PERFORMANCE_EVERY)
            + "    "
            + "N_RUNS_STATS="
            + str(N_RUNS_STATS)
    )
    return subtitle


# Init env ________________________________________________________________
def autoset_numerical_parameters():
    """
    Autoset parameters that have not been set yet.

    Returns:
        N (int): linear grid size
        Nhits (int): max number of hits
        stop_t (int): max search duration
        Nruns (int): number of episodes for computing stats
    """
    testenv = env(
        Ndim=N_DIMS,
        lambda_over_dx=LAMBDA_OVER_DX,
        R_dt=R_DT,
        norm_Poisson=NORM_POISSON,
        Ngrid=N_GRID,
        Nhits=N_HITS,
        initial_hit=None,
        dummy=True,
    )
    if STOP_t is None:
        if N_DIMS == 1:
            stop_t = int(round(4 * testenv.N))
        else:
            if testenv.mu0_Poisson < 1e-3:
                stop_t = 10 * testenv.N ** N_DIMS
            elif testenv.mu0_Poisson < 1:
                stop_t = int(round(5 * 10 ** N_DIMS * LAMBDA_OVER_DX / np.sqrt(testenv.mu0_Poisson)))
            else:
                stop_t = int(round(5 * 10 ** N_DIMS * LAMBDA_OVER_DX))
    else:
        stop_t = STOP_t

    if N_RUNS_STATS is None:
        if N_DIMS == 1:
            Nruns = 5000
        elif N_DIMS == 2:
            Nruns = 3200
        else:
            raise Exception("Nruns not pre-defined for N_DIMS > 2")
    else:
        Nruns = N_RUNS_STATS

    return testenv.N, testenv.Nhits, stop_t, Nruns


def init_env():
    """
    Instanciate an environment.

    Returns:
        env (SourceTracking): instance of the source-tracking POMDP
    """
    myenv = env(
        Ndim=N_DIMS,
        lambda_over_dx=LAMBDA_OVER_DX,
        R_dt=R_DT,
        norm_Poisson=NORM_POISSON,
        Ngrid=N_GRID,
        Nhits=N_HITS,
        initial_hit=None,
    )

    return myenv


# Compute, print and plot stats ________________________________________________________________
def compute_stats(Nepisodes, policy, parallel=True):
    """
    Launches the episodes and save statistics in a dictionary.

    Args:
        Nepisodes (int):
            number of episodes to compute the statistics
        policy (int):
            policy to evaluate
        parallel (bool, optional):
            whether to run the episodes in parallel (warning: code may hang for large models) (default=True)

    Returns:
        stats (dict): contains 'p_not_found', 'mean', 'std', ...
    """
    # Running the episodes
    if parallel:    # parallel, will hang for large NN
        pool = multiprocessing.Pool(N_PARALLEL)
        episodes = range(Nepisodes)
        inputargs = zip(episodes, repeat(policy, Nepisodes))
        pdfs_t, _, _ = zip(*pool.starmap(WorkerStats, inputargs))
        pool.close()
        pool.join()
    else:   # sequential
        pdfs_t = []
        for episode in range(Nepisodes):
            a, _, _ = WorkerStats(episode, policy)
            pdfs_t.append(a)

    # Compiling the results
    pdfs_t = np.asarray(pdfs_t)

    # Individual stats to estimate 95 % confidence interval
    bin_t = np.arange(pdfs_t.shape[1])
    tots = np.sum(pdfs_t, axis=1)
    means = np.sum(pdfs_t * bin_t, axis=1) / tots
    # margin of error for 95 % CI
    t_val = 1.96  # assumes N_RUNS > 200
    N_runs = len(means)
    mean_err = t_val * np.std(means) / np.sqrt(N_runs)

    # Reduce
    pdf_t = np.mean(pdfs_t, axis=0)
    pdf_t = np.trim_zeros(pdf_t, trim='b')

    cdf_t = np.cumsum(pdf_t)
    bin_t = np.arange(len(pdf_t))
    tot = np.sum(pdf_t)
    mean = np.sum(pdf_t / tot * bin_t)
    std = np.sqrt(np.sum(pdf_t / tot * bin_t ** 2) - mean ** 2)
    p50 = np.interp(0.5, cdf_t, bin_t, left=np.nan, right=np.nan)
    p99 = np.interp(0.99, cdf_t, bin_t, left=np.nan, right=np.nan)

    stats = {
        'policy': policy,
        'p_not_found': 1.0 - tot,
        'mean': mean,
        'mean_err': mean_err,
        'std': std,
        'p50': p50,
        'p99': p99,
        'pdf': pdf_t,
    }

    return stats


def print_stats(stats1, stats2=None):
    """
    Print stats on screen.

    Args:
        stats1 (dict): stats of the policy (as computed by :func:`compute_stats`)
        stats2 (dict or None, optional): stats of another policy (as computed by :func:`compute_stats`) for comparison
    """
    vars = ['p_not_found', 'mean', 'std', 'p50', 'p99']
    for var in vars:
        if stats2 is None:
            if var == 'p_not_found':
                print(var, "\t", "{:.4e}".format(stats1[var]))
            else:
                print(var, "\t\t", "{:.4f}".format(stats1[var]))
        else:
            if var == 'p_not_found':
                print(var, "\t", "{:.4e}".format(stats1[var]), "\t\tref: ", "{:.4e}".format(stats2[var]))
            else:
                print(var, "\t\t", "{:.4f}".format(stats1[var]), "\t\tref: ", "{:.4f}".format(stats2[var]))


def plot_stats(statsRL, statsref=None, title='', file_suffix='0'):
    """
    Plot performance stats of the RL policy in a figure and save it.

    Args:
        statsRL (dict): stats of the RL policy (as computed by :func:`compute_stats`)
        statsref (dict or None, optional): stats of reference policy (as computed by :func:`compute_stats`)
        title (str, optional): plot title (printed on top)
        file_suffix (str, optional): suffix for file name, to prevent overwriting
    """

    fig_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_figure_stats_" + str(file_suffix) + ".png"))
    fig, ax = plt.subplots(1, 3, figsize=(20, 7.5))
    plt.subplots_adjust(left=0.05, bottom=0.24, right=0.99, top=0.9, wspace=0.15, hspace=0)
    palette = plt.get_cmap('tab10')
    for k in range(2):
        if k == 0:
            stats = statsRL
        elif k == 1:
            if statsref is None:
                continue
            stats = statsref
        policy = stats["policy"]
        if policy == -1:
            label = "RL"
        else:
            label = policy_name(policy)
        color = palette(k)
        kwargs0 = {'markersize': 5 / (k+1), 'linewidth': 2 / (k+1), 'color': palette(k)}
        kwargs1 = {'fontsize': 10, 'color': color, 'xycoords': 'axes fraction', 'ha': "right"}
        for i in range(3):
            if i == 0:
                yvar = stats['pdf']
                yname = "PDF"
            elif i == 1:
                yvar = np.cumsum(stats['pdf'])
                yname = "CDF"
            else:
                yvar = 1.0 - np.cumsum(stats['pdf'])
                yname = "CCDF (=1-CDF)"
            x = np.arange(len(yvar))
            if i < 2:
                line, = ax[i].plot(x, yvar, '-o', label=label, **kwargs0)
            else:
                line, = ax[i].semilogy(x, yvar, '-o', label=label, **kwargs0)

            ax[i].set_title(yname + " of number of steps")
            ax[i].set_xlabel("number of steps")
            if i == 0:
                ax[i].annotate("p_not_found = " + "{:.3e}".format(stats['p_not_found']), (0.95, 0.85-0.05*k), **kwargs1)
                ax[i].annotate("mean = " + "{:.3e}".format(stats['mean']), (0.95, 0.65-0.05*k), **kwargs1)
                ax[i].annotate("std = " + "{:.3e}".format(stats['std']), (0.95, 0.45-0.05*k), **kwargs1)
            elif i == 1:
                ax[i].annotate("P50 = " + "{:.3e}".format(stats['p50']), (0.95, 0.45-0.05*k), **kwargs1)
                ax[i].annotate("P99 = " + "{:.3e}".format(stats['p99']), (0.95, 0.30-0.05*k), **kwargs1)
                ax[i].legend(fontsize=10, loc='lower center')

    fig.suptitle(title, y=0.98)
    plt.figtext(0.5, 0.003, param2subtitle(MYENV, MYMODEL), fontsize=7, ha="center", va="bottom")
    plt.draw()
    fig.savefig(fig_file)
    plt.close(fig)


def plot_stats_evolution(data, ref_stats=None, title=''):
    """
    Plot performance vs training time.

    Args:
        data (ndarray): contains the history of performance
            Each row corresponds to a training iteration, and the columns contain

                - col 0: training iteration
                - col 1: number of transitions seen
                - col 2: number of transitions generated
                - col 3: epsilon from epsilon-greedy exploration
                - col 4: probability that the source is never found
                - col 5: mean number of steps to find the source, provided that the source is ultimately found
                - col 6: error on the mean
                - col 7: number of steps to find the source with 50 % probability
                - col 8: number of steps to find the source with 99 % probability

        ref_stats (dict, optional): stats of the reference policy (as computed by :func:`compute_stats`) for comparison
        title (str, optional): a title for the plot
    """
    # data columns: 0=it, 1=Ntrans_seen, 2=N_trans_gen, 3=eps, 4=p_not_found, 5=mean, 6=mean_err, 7=p50, 8=p99
    index_list = [3, 4, 5, 5, 7, 8]
    names_list = ("eps", "p_not_found", "mean", "rel_mean", "p50", "p99")

    fig_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_figure_learning_progress" + ".png"))
    fig_file_bkp = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_figure_learning_progress" + "_bkp.png"))
    if os.path.isfile(fig_file):
        os.rename(fig_file, fig_file_bkp)

    ncols = len(index_list)
    fig, ax = plt.subplots(2, ncols, figsize=(max(6 * ncols, 20), 13))
    plt.subplots_adjust(left=0.03, bottom=0.16, right=0.99, top=0.92, wspace=0.2, hspace=0.3)

    color = "tab:blue"
    colorref = "tab:orange"

    for r in range(2):
        if r == 0:
            xvar = data[:, 0]
            xlab = "training iterations"
            xmax = data[-1, 0]
        elif r == 1:
            xvar = data[:, 1]
            xlab = "number of transitions seen"
            xmax = data[-1, 1]

        for i in range(ncols):
            yvar = data[:, index_list[i]]
            if names_list[i] == "rel_mean":
                if ref_stats is not None:
                    yref0 = ref_stats["mean"]
                else:
                    yref0 = np.nan
                yvar = yvar/yref0
            if names_list[i] == "p_not_found":
                ax[r, i].semilogy(xvar, yvar, '-o', markersize=4, color=color)
            else:
                line, = ax[r, i].plot(xvar, yvar, '-o', markersize=4, color=color, label="RL")
            ax[r, i].set_xlabel(xlab)
            ax[r, i].set_title(names_list[i])
            ax[r, i].set_xlim(0, max(xmax, 1))

            # for mean and rel_mrean
            if names_list[i] in ("mean", "rel_mean"):
                # add confidence interval on the mean
                err = data[:, 6]
                if names_list[i] == "rel_mean":
                    err = err/yref0
                ax[r, i].fill_between(xvar, yvar - err, yvar + err, color=color, alpha=0.2)  # 95 % confidence interval
                # add where p_not_found is too high
                flag = data[:, 4] > 1e-3
                ax[r, i].plot(xvar[flag], yvar[flag], 'x', markersize=7, markeredgewidth=2, color=color)

            # add baseline on the same plot
            if (ref_stats is not None) and (names_list[i] in ("p_not_found", "mean", "p50", "p99")):
                yref = ref_stats[names_list[i]] * np.ones(len(yvar))
                if names_list[i] == "p_not_found":
                    ax[r, i].semilogy(xvar, yref, '--', color=colorref)
                else:
                    line, = ax[r, i].plot(xvar, yref, '--', color=colorref, label=policy_name(ref_stats["policy"]))
                # for mean, add confidence interval
                if names_list[i] == "mean":
                    err = ref_stats["mean_err"] * np.ones(len(yvar))
                    ax[r, i].fill_between(xvar, yref - err, yref + err, color=colorref, alpha=0.2)  # 95 % CI

            # legend
            if names_list[i] == "mean":
                ax[r, i].legend(fontsize=12)

    fig.suptitle(title, y=0.98)
    plt.figtext(0.5, 0.003, param2subtitle(MYENV, MYMODEL), fontsize=7, ha="center", va="bottom")
    plt.draw()
    fig.savefig(fig_file)
    plt.close(fig)
    if os.path.isfile(fig_file_bkp):
        os.remove(fig_file_bkp)

# Worker (compute trajectories) _______________________________________________________________
def WorkerStats(episode, policy):
    """Wrapper of Worker for computing stats of a policy."""
    pdf_t, _, _ = _Worker(episode=episode, policy=policy, eps=0.0, memorize=False)
    return pdf_t, _, _

def WorkerExp(episode, policy, eps):
    """Wrapper of Worker for generating new experience."""
    pdf_t, memory_s, memory_sp = _Worker(episode=episode, policy=policy, eps=eps, memorize=True)
    return pdf_t, memory_s, memory_sp

def _Worker(episode, policy, eps, memorize):
    """Execute one episode.

    1. The agent is placed in the center of the grid world
    2. The agent receives an initial hit
    3. The agent moves according to the model
    4. The episode terminates when the agent has almost certainly found the source

    Args:
        episode (int): episode ID
        policy (int): policy to follow
        eps (0 <= float <= 1): value of eps for exploration
        memorize: whether to memorize states s and s' along the search

    Returns:
        pdf_t (numpy array): distribution of the number of steps to find the source
        memory_s (numpy array or empty): memory of t-states (s), if memorize = True
        memory_sp (numpy array or empty): memory of (t+1)-states (s'), if memorize = True
    """
    # copy the env to save time, and reset (according to randomly drawn hit)
    myenv = deepcopy(MYENV)
    myenv.restart()
    if policy != -1:
        mypol = HeuristicPolicy(env=myenv, policy=policy)

    # Initialization parameters
    p_not_found_yet = 1  # probability of not having found the source
    pdf_t = np.zeros(STOP_t)
    t = 0
    stop = 0
    memory_s, memory_sp = [], []

    # main loop
    while True:

        # compute possible next states for all actions/hits
        if SYM_TRAIN_RANDOMIZE and memorize:
            if myenv.Ndim > 2:
                raise Exception("SYM_TRAIN_RANDOMIZE only implemented in 1D and 2D")
            if myenv.Ndim == 1:
                Nsym = 2
            elif myenv.Ndim == 2:
                Nsym = 8
            sym = np.random.RandomState().choice(Nsym)
            if sym != 0:
                myenv.apply_sym_transformation(sym)  # transforms p_source, state, agent, ...
        state, statep = myenv.transitions()  # 1, (Nactions, Nhits)

        if random.random() < eps:  # epsilon-greedy
            action = random.randint(0, myenv.Nactions - 1)
        else:
            if policy == -1:
                action = myenv.choose_action_from_statep(
                    model=MYMODEL,
                    statep=statep,
                    sym_avg=((not memorize) and SYM_EVAL_ENSEMBLE_AVG)
                )
            else:
                action = mypol.choose_action()

        #  adding the new experience to memory
        if memorize:
            memory_s.append(state)
            memory_sp.append(statep)

        # step in myenv
        hit, p_end, _ = myenv.step(action, quiet=True)

        # memory_ndim p
        t += 1
        pdf_t[t] = p_not_found_yet * p_end

        # Has the source been found?
        if p_not_found_yet < STOP_p or p_end > 1 - EPSILON:
            stop = 1
        elif t >= STOP_t - 1:
            stop = 2
        elif myenv.agent_stuck:
            stop = 3

        if stop:
            if memorize:
                memory_s = np.asarray(memory_s)  # (Nsteps)
                memory_sp = np.asarray(memory_sp)  # (Nsteps, Nactions, Nhits)
                    
            break

        p_not_found_yet *= 1 - p_end

    return pdf_t, memory_s, memory_sp


# DQN utils _______________________________________________________________
def new_experience(Nepisodes, policy, eps, parallel):
    """
    Generate trajectories according to policy with eps-exploration

    Args:
        Nepisodes (int): number of episodes to simulate
        policy (int): policy used by the agent in the episodes
        eps (float): a random action is chosen with probability eps
        parallel (bool): whether episodes are computed in parallel

    Returns:
        states (ndarray): array of states s
        statesp (ndarray): array of corresponding successor states s'
    """
    # Running the episodes
    if parallel:  # parallel, but may hang for large NN
        pool = multiprocessing.Pool(N_PARALLEL)
        episodes = range(Nepisodes)
        inputargs = zip(episodes, repeat(policy, Nepisodes), repeat(eps, Nepisodes))
        _, memory_states, memory_statesp = zip(*pool.starmap(WorkerExp, inputargs))
        pool.close()
        pool.join()
    else:  # sequential
        memory_states = []
        memory_statesp = []
        for episode in range(Nepisodes):
            _, b, c = WorkerExp(episode, policy, eps)
            memory_states.append(b)
            memory_statesp.append(c)

    # Reshaping simulation data
    states = np.empty(tuple([0] + list(memory_states[0].shape[1:])))        # (Nsteps)
    statesp = np.empty(tuple([0] + list(memory_statesp[0].shape[1:])))      # (Nsteps, Nactions, Nhits)
    for s, sp in zip(memory_states, memory_statesp):
        states = np.concatenate((states, s), axis=0)
        statesp = np.concatenate((statesp, sp), axis=0)

    return states, statesp


def update_buffer_memory(mem, new, max_size):
    """
    Add new transitions to the memory buffer.

    Args:
        mem (ndarray): current memory buffer
        new (ndarray): array of new transitions
        max_size (int): max number of transitions in memory

    Returns:
        mem (ndarray): updated memory
    """
    mem_size = mem.shape[0]
    new_size = new.shape[0]
    if new_size > max_size:
        raise Exception("Memory is too small to add this many new episodes!")
    if mem_size + new_size <= max_size:
        mem = np.concatenate((mem, new), axis=0)
    else:
        delete_size = mem_size + new_size - max_size
        mem = np.concatenate((mem[delete_size:], new), axis=0)

    assert mem.shape[0] <= max_size

    return mem


#  Training core algo _______________________________________________________________
def train_model(eps_floor, eps_0, eps_decay, max_it, ref_stats):
    """
    Train the model to fit an (approximately) optimal value function using a model-based version of DQN
    (Mnih et al., Nature 2015).

    The value model is trained using simulation data generated by the RL policy with eps-greedy exploration
    (a random action is selected with eps probability, which decreases over time).

    The loss is the norm of the Bellman optimality error (the norm used is defined within the model).

    Training can be interrupted at any time (the model is saved periodically).

    Args:
        eps_floor (float): floor value of the exploration parameter eps
        eps_0 (float): initial value of the exploration parameter eps
        eps_decay (int): decay timescale of the exploration parameter eps (in number of training iterations)
        max_it (int): training stops if training iteration > max_it
        ref_stats (dict or None): stats of a reference policy (as computed by :func:`compute_stats`), for plots only
    """
    def eps_exploration(iteration, e_floor, e_0, e_decay):
        if e_decay is None:
            return e_floor
        else:
            return max(e_0 * np.exp(-iteration / e_decay), e_floor)

    def get_target(statep):
        if DDQN:
            target = MYENV.get_target(modelvalue=MYFROZENMODEL, modelaction=MYMODEL, statep=statep)
        else:
            target = MYENV.get_target(modelvalue=MYFROZENMODEL, modelaction=None, statep=statep)
        return target

    print("* populating memory...")
    # before training starts, populate ~ 90 % of memory with data from initial policy
    Nepisodes = max(int(0.9 * MEMORY_SIZE / STOP_t), 1)
    print("Nepisodes: ", Nepisodes)
    eps = eps_exploration(0, e_floor=eps_floor, e_0=eps_0, e_decay=eps_decay)
    print("eps: ", eps)
    states, statesp = new_experience(Nepisodes=Nepisodes, policy=-1, eps=eps, parallel=Nepisodes >= N_PARALLEL > 1)
    print("occupied memory: ", states.shape[0], "/", MEMORY_SIZE)
    avg_len_episode = states.shape[0] / Nepisodes
    if states.shape[0] > MEMORY_SIZE:
        states = states[:MEMORY_SIZE]
        statesp = statesp[:MEMORY_SIZE]

    # init input shapes
    _ = MYENV.get_state_value(MYMODEL, states[0])

    # iterating
    print("* start training...")
    it = 0
    stats_history = np.nan * np.zeros(9)

    N_transitions_generated = states.shape[0]
    N_transitions_seen = 0

    while it <= max_it:

        ###################### SAVING

        # *** Save model
        if it % SAVE_MODEL_EVERY == 0:
            model_name = str(RUN_NAME + '_model')
            model_path = os.path.join(DIR_MODELS, model_name)
            MYMODEL.save_model(model_path)

        if it % EVALUATE_PERFORMANCE_EVERY == 0:
            model_name = str(RUN_NAME + '_model_bkp_' + str(it // EVALUATE_PERFORMANCE_EVERY))
            model_path = os.path.join(DIR_MODELS, model_name)
            MYMODEL.save_model(model_path)

        ###################### EVALUATE PERFORMANCE OF CURRENT MODEL

        # *** Generate accurate stats of current model-based policy, plot the results
        if it % EVALUATE_PERFORMANCE_EVERY == 0:
            print("* evaluating performance of the current model-based policy...")
            stats = compute_stats(Nepisodes=N_RUNS_STATS, policy=-1, parallel=N_RUNS_STATS >= N_PARALLEL > 1)
            print_stats(stats, ref_stats)
            titlestr = "evaluation of current RL policy, training it = " + str(it) \
                       + ", transitions seen = " + "{:.2e}".format(N_transitions_seen)
            plot_stats(statsRL=stats,
                       statsref=ref_stats,
                       title=titlestr,
                       file_suffix=str(it // EVALUATE_PERFORMANCE_EVERY))
            add_stats = np.array([it, N_transitions_seen, N_transitions_generated, eps,
                                  stats["p_not_found"], stats["mean"], stats["mean_err"], stats["p50"], stats["p99"]])
            stats_history = np.vstack((stats_history, add_stats))
            stats_history_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_table_stats" + ".npy"))
            np.save(stats_history_file, stats_history)
            titlestr = "evolution of performance during training (currently: training it = " + str(it) \
                       + ", transitions seen = " + "{:.2e}".format(N_transitions_seen) + ")"
            plot_stats_evolution(data=stats_history,
                                 ref_stats=ref_stats,
                                 title=titlestr)
            print(">>> Current results saved in the directory: " + DIR_OUTPUTS)

        ###################### GENERATE EXP

        # *** Generate new experience and add it to buffer memory
        eps = eps_exploration(it, e_floor=eps_floor, e_0=eps_0, e_decay=eps_decay)
        # compute new trajectories
        Nepisodes = max(int(np.ceil(NEW_TRANS_PER_IT / avg_len_episode)), 1)
        new_states, new_statesp = new_experience(Nepisodes=Nepisodes,
                                                 policy=-1,
                                                 eps=eps,
                                                 parallel=Nepisodes >= N_PARALLEL > 1)
        avg_len_episode = new_states.shape[0] / Nepisodes
        if new_states.shape[0] > MEMORY_SIZE:
            print("Memory size is too small to fit this many new transitions; it = ", it, ", Nepisodes = ", Nepisodes,
                  ", new = ", new_states.shape[0], ", mem = ", MEMORY_SIZE)
            new_states = new_states[:MEMORY_SIZE]
            new_statesp = new_statesp[:MEMORY_SIZE]
        # delete oldest experiences and add the new ones to memory
        states = update_buffer_memory(states, new_states, max_size=MEMORY_SIZE)
        statesp = update_buffer_memory(statesp, new_statesp, max_size=MEMORY_SIZE)
        renewed_mem = new_states.shape[0] / MEMORY_SIZE
        N_transitions_generated += new_states.shape[0]

        ###################### TRAINING

        # update model used for computing targets
        if it % UPDATE_FROZEN_MODEL_EVERY == 0:
            MYFROZENMODEL.set_weights(MYMODEL.get_weights())

        # *** Perform step(s) of mini-batch gradient descent
        mean_loss = 0
        memory_size = states.shape[0]
        for gd_step in range(N_GD_STEPS):
            batch = np.random.randint(memory_size, size=BATCH_SIZE)
            x, _, = MYENV.states2inputs(states[batch], 0)  # (BATCH_SIZE , inputshape), (BATCH_SIZE)
            y = get_target(statesp[batch])
            loss = MYMODEL.train_step(x, y, augment=SYM_TRAIN_ADD_DUPLICATES)  # train (model weights are updated)
            mean_loss += loss
        mean_loss /= N_GD_STEPS

        N_transitions_seen += N_GD_STEPS * BATCH_SIZE

        it += 1

        ###################### MONITORING

        # *** Print info to screen
        if it % PRINT_INFO_EVERY == 0:

            print(
                "---- training iteration:", it,
                "  |  eps:", eps,
                "  |  episodes added:", Nepisodes,
                "  |  fraction memory renewed:", renewed_mem,
                "  |  transitions seen:", N_transitions_seen,
                "  |  loss:", float(mean_loss),
                "  |  ", time.strftime("%Y-%m-%d %H:%M:%S")
            )

    print("Stopped: max number of training iterations reached.")


# Main func  _______________________________________________________________
def run():
    """Main function that calls the training algorithm and evaluate its performance"""
    # Print and save parameters
    print("\n*** Parameters summary...")
    save_parameters(MYENV, MYMODEL)

    if POLICY_REF is not None:
        # Infotaxis stats, for reference
        print("\n*** Computing stats of reference policy (" + str(POLICY_REF) + ") for comparison...")
        ref_stats = compute_stats(Nepisodes=N_RUNS_STATS, policy=POLICY_REF, parallel=True)
        print_stats(ref_stats)
        if NEW_TRANS_PER_IT < ref_stats['mean']:
            raise Exception("Increase NEW_TRANS_PER_IT (ie increase BATCH_SIZE or N_GD_STEPS)")
    else:
        ref_stats = None

    # Training
    print("\n*** Learning optimal value function...")
    train_model(
        eps_floor=E_GREEDY_FLOOR,
        eps_0=E_GREEDY_0,
        eps_decay=E_GREEDY_DECAY,
        max_it=ALGO_MAX_IT,
        ref_stats=ref_stats,
    )

    # Compute stats of the RL policy
    print("\n*** Computing stats of the RL policy...")
    stats = compute_stats(Nepisodes=N_RUNS_STATS, policy=-1, parallel=N_RUNS_STATS >= N_PARALLEL > 1)

    print_stats(stats, ref_stats)
    plot_stats(statsRL=stats,
               statsref=ref_stats,
               title="evaluation of RL policy",
               file_suffix='evaluation')


if __name__ == '__main__':

    if not os.path.isdir(DIR_OUTPUTS):
        os.makedirs(DIR_OUTPUTS)

    if not os.path.isdir(DIR_MODELS):
        os.makedirs(DIR_MODELS)

    print("\n*** Autoset params...")
    N_GRID, N_HITS, STOP_t, N_RUNS_STATS = autoset_numerical_parameters()
    print("N_GRID =", N_GRID, " N_HITS =", N_HITS, " N_RUNS_STATS =", N_RUNS_STATS, " MEMORY_SIZE =", MEMORY_SIZE)

    NEW_TRANS_PER_IT = int(BATCH_SIZE * N_GD_STEPS / REPLAY_NTIMES)
    if NEW_TRANS_PER_IT > 0.8 * MEMORY_SIZE:
        print("Nb of new transitions per it (approx): ", NEW_TRANS_PER_IT)
        print("Memory size:", MEMORY_SIZE)
        raise Exception("Memory is too small for these BATCH_SIZE, N_GD_STEPS and REPLAY_NTIMES")
    elif NEW_TRANS_PER_IT < 1:
        raise Exception("Not enough new transitions per it, increase BATCH_SIZE or N_GD_STEPS")

    print("\n*** Building env...")
    MYENV = init_env()

    # Model initialization
    print("\n*** Building model...")
    MYMODEL = build_new_model(
            Ndim=N_DIMS,
            FC_layers=FC_LAYERS,
            FC_units=FC_UNITS,
            learning_rate=LEARNING_RATE,
        )
    MYFROZENMODEL = build_new_model(
            Ndim=N_DIMS,
            FC_layers=FC_LAYERS,
            FC_units=FC_UNITS,
            learning_rate=LEARNING_RATE,
        )
    MYFROZENMODEL.set_weights(MYMODEL.get_weights())

    if CONTINUE_TRAINING:
        print("\n*** Reloading previous weights from:", MODEL_PATH)
        oldmodel = reload_model(MODEL_PATH, inputshape=MYENV.NN_input_shape)
        MYMODEL.set_weights(oldmodel.get_weights())
        MYFROZENMODEL.set_weights(oldmodel.get_weights())
        del oldmodel

    # Main program
    run()




