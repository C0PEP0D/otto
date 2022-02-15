#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to evaluate the performance of a given policy (such as intotaxis or an RL policy)
on the source-tracking POMDP.

The script records many statistics and monitoring information, and plot results.

Computations are parallelized with ``multiprocessing`` (default) or with MPI (requires the ``mpi4py`` module).

To use MPI on n_cpus processors, run the following command line from a Linux terminal:

``mpiexec -n n_cpus python3 -m mpi4py evaluate.py -i custom_params.py``

The list of all parameters is given below.

Source-tracking POMDP
    - N_DIMS (int > 0)
        number of space dimensions (1D, 2D, ...)
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

Policy
    - POLICY (int)
        - -1: reinforcement learning
        - 0: infotaxis (Vergassola, Villermaux and Shraiman, Nature 2007)
        - 1: space-aware infotaxis
        - 2: custom policy (to be implemented by the user)
        - 5: random walk
        - 6: greedy policy
        - 7: mean distance policy
        - 8: voting policy (Cassandra, Kaelbling & Kurien, IEEE 1996)
        - 9: most likely state policy (Cassandra, Kaelbling & Kurien, IEEE 1996)
    - STEPS_AHEAD (int >= 1)
        number of anticipated moves, can be > 1 only for POLICY=0

Statistics computation
    - ADAPTIVE_N_RUNS (bool)
        if True, more episodes will be simulated until the estimated error is less than REL_TOL
    - REL_TOL (0.0 < float < 1.0)
        if ADAPTIVE_N_RUNS: tolerance on the relative error on the mean number of steps to find the source
    - MAX_N_RUNS (int > 0 or None)
        if ADAPTIVE_N_RUNS: maximum number of runs, set automatically if None
    - N_RUNS (int > 0 or None)
        if not ADAPTIVE_N_RUNS: number of episodes to simulate, set automatically if None
    - STOP_p (float ~ 0.0)
        episode stops when the probability that the source has been found is greater than 1 - STOP_p

Saving
    - RUN_NAME (str or None)
        prefix used for all output files, if None will use a timestamp

Parallelization
    - N_PARALLEL (int)
        number of episodes computed in parallel (if <= 0, will use all available cpus)

        This is only when using multiprocessing for parallelization (it has no effect with MPI).

        Known bug: for large neural networks, the code may hang if N_PARALLEL > 1, so use N_PARALLEL = 1 instead.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], '..', '..')))
sys.path.insert(2, os.path.abspath(os.path.join(sys.path[0], '..', '..', 'zoo')))
import time
import argparse
import importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import skew as Skew
from scipy.stats import kurtosis as Kurt
from otto.classes.sourcetracking import SourceTracking as env
if 'mpi4py' in sys.modules:
    WITH_MPI = True
    import mpi4py.MPI as MPI
else:
    WITH_MPI = False
    import multiprocessing

# import default globals
from otto.evaluate.parameters.__defaults import *

# import globals from user defined parameter file
if os.path.basename(sys.argv[0]) not in ["sphinx-build", "build.py"]:
    parser = argparse.ArgumentParser(description='Evaluate a policy')
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

# set other globals
if POLICY == -1:
    from otto.classes.rlpolicy import RLPolicy
    from otto.classes.valuemodel import reload_model
    if MODEL_PATH is None:
        raise Exception("MODEL_PATH cannot be None with an RL policy!")
else:
    from otto.classes.heuristicpolicy import HeuristicPolicy

EPSILON = 1e-10

if RUN_NAME is None:
    RUN_NAME = time.strftime("%Y%m%d-%H%M%S")

DIR_OUTPUTS = os.path.abspath(os.path.join(sys.path[0], "outputs", RUN_NAME))
PARENT_DIR_TMP = os.path.abspath(os.path.join(sys.path[0], "tmp"))
DIR_TMP = os.path.abspath(os.path.join(PARENT_DIR_TMP, RUN_NAME))


# config
matplotlib.use('Agg')

# _______________________________________________

def am_i_root():
    """Returns true if master process, false otherwise"""
    if WITH_MPI:
        return not ME
    else:
        return os.getpid() == MASTER_PID


def autoset_numerical_parameters():
    """
    Autoset parameters that have not been set yet.

    Returns:
        N (int): linear grid size
        Nhits (int): max number of hits
        stop_t (int): max search duration
        Nruns (int): number of episodes for computing stats (initial value)
        max_Nruns (int): hard limit on the number of episodes
        mu0_Poisson (float): physical parameter derived from lambda_over_dx and R_dt
    """
    testenv = env(
        Ndim=N_DIMS,
        lambda_over_dx=LAMBDA_OVER_DX,
        R_dt=R_DT,
        norm_Poisson=NORM_POISSON,
        Ngrid=N_GRID,
        Nhits=N_HITS,
        dummy=True,
    )
    if N_DIMS == 1:
        stop_t = int(round(4 * testenv.N))
    else:
        if testenv.mu0_Poisson < 1e-3:
            stop_t = 10 * testenv.N ** N_DIMS
        elif testenv.mu0_Poisson < 1:
            stop_t = int(round(5 * 10 ** N_DIMS * LAMBDA_OVER_DX / np.sqrt(testenv.mu0_Poisson)))
        else:
            stop_t = int(round(5 * 10 ** N_DIMS * LAMBDA_OVER_DX))

    if N_RUNS is None:
        # predefined for REL_TOL = 0.01
        if N_DIMS == 1:
            Nruns = 16000
        elif N_DIMS == 2:
            Nruns = 6400
        elif N_DIMS == 3:
            Nruns = 25600
        elif N_DIMS == 4:
            Nruns = 102400
        else:
            raise Exception("Nruns not pre-defined for N_DIMS > 4")
        Nruns = int(Nruns * (0.01 / REL_TOL) ** 2)
    else:
        Nruns = N_RUNS

    if MAX_N_RUNS is None:
        max_Nruns = MAX_N_RUNS
    else:
        max_Nruns = 10 * Nruns

    if ADAPTIVE_N_RUNS or WITH_MPI:
        Nruns = int(N_PARALLEL * (np.ceil(Nruns / N_PARALLEL)))  # make it multiple of N_PARALLEL
        max_Nruns = int(N_PARALLEL * (np.ceil(max_Nruns / N_PARALLEL)))  # make it multiple of N_PARALLEL

    return testenv.N, testenv.Nhits, stop_t, Nruns, max_Nruns, testenv.mu0_Poisson


def init_envs():
    """
    Instanciate an environment and a policy.

    Returns:
        myenv (SourceTracking): instance of the source-tracking POMDP
        mypol (Policy): instance of the policy
    """
    myenv = env(
        Ndim=N_DIMS,
        lambda_over_dx=LAMBDA_OVER_DX,
        R_dt=R_DT,
        norm_Poisson=NORM_POISSON,
        Ngrid=N_GRID,
        Nhits=N_HITS,
    )
    if POLICY == -1:
        mymodel = reload_model(MODEL_PATH, inputshape=myenv.NN_input_shape)
        mypol = RLPolicy(
            env=myenv,
            model=mymodel,
        )
    else:
        mypol = HeuristicPolicy(
            env=myenv,
            policy=POLICY,
            steps_ahead=STEPS_AHEAD,
        )
    return myenv, mypol


def check_envs(env, pol):
    """Check that the attributes of env and pol match the required parameters."""
    assert env.Ndim == N_DIMS
    assert env.lambda_over_dx == LAMBDA_OVER_DX
    assert env.R_dt == R_DT
    assert env.mu0_Poisson == MU0_POISSON
    assert env.norm_Poisson == NORM_POISSON
    assert env.N == N_GRID
    assert env.Nhits == N_HITS
    assert not env.draw_source
    
    assert pol.env == env
    assert pol.policy_index == POLICY
    if POLICY != -1:
        assert pol.steps_ahead == STEPS_AHEAD
    

def stats_from_pdf(x, pdf_x):
    """Compute mean, standard deviation, skewness, kurtosis and norm from a pdf
    (probability distribution function)."""
    mean = 0.0
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    tot = 0.0
    for bin in range(len(pdf_x)):
        tot += pdf_x[bin]
        mean += x[bin] * pdf_x[bin]
        m2 += x[bin] ** 2 * pdf_x[bin]
        m3 += x[bin] ** 3 * pdf_x[bin]
        m4 += x[bin] ** 4 * pdf_x[bin]
    mean /= tot
    m2 /= tot
    m3 /= tot
    m4 /= tot
    sigma = np.sqrt(m2 - mean ** 2)
    skew = (m3 - 3 * mean * m2 + 2 * mean ** 3) / (sigma ** 3)
    kurt = (m4 - 4 * mean * m3 + 6 * mean ** 2 * m2 - 3 * mean ** 4) / (sigma ** 4) - 3.0
    return mean, sigma, skew, kurt, tot


def stats_from_cdf(x, cdf_x):
    """Compute time to find the source with some % probability from a cdf (cumulative distribution function)."""
    percentiles = []
    for P in (0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        b = 0
        while cdf_x[b] < P:
            b += 1
            if b >= len(cdf_x):
                break
        if b >= len(cdf_x):
            pP = np.nan
        else:
            pP = x[b - 1] + (P - cdf_x[b - 1]) / (cdf_x[b] - cdf_x[b - 1]) * (x[b] - x[b - 1])
        percentiles.append(pP)
    percentiles.append(cdf_x[-1])
    return tuple(percentiles)


def cdf_to_pdf(cdf):
    """Compute a pdf (probability distribution function) from its cdf (cumulative distribution function)."""
    pdf = deepcopy(cdf)
    pdf[1:] -= pdf[:-1].copy()
    return pdf


def Worker(episode):
    """
    Execute one episode.

    1. The agent is placed in the center of the grid world
    2. The agent receives a random initial hit
    3. The agent moves according to the policy
    4. The episode terminates when the agent has almost certainly found the source

    Args:
        episode (int): episode ID

    Returns:
        cdf_t (ndarray): CDF (cumulative distribution function) of the number of steps to find the source
        cdf_h (ndarray): CDF (cumulative distribution function) of the number of cumulated hits to find the source
        T_mean (float): mean number of steps to find the source in this episode
        failed (bool): whether the episode ended due to failure (such as infinite loop or timeout)
    """
    episode_start_time = time.monotonic()

    stop = 0
    near_boundaries = 0
    failed = 0

    cdf_t = -1.0 * np.ones(LEN_CDF_T)
    cdf_h = -1.0 * np.ones(LEN_CDF_H)
    bin_t = 0
    bin_h = 0
    cdf_t[bin_t] = 0.0
    cdf_h[bin_h] = 0.0

    myenv, mypol = init_envs()
    check_envs(myenv, mypol)

    t = 0  # time step
    cumulative_hits = 0  # cumulative number of hits (int)
    T_mean = 0
    p_not_found_yet = 1

    while True:

        # choice of action
        action = mypol.choose_action()

        # step in myenv
        hit, p_end, done = myenv.step(action)
        near_boundaries = max(near_boundaries, myenv.agent_near_boundaries)

        t += 1
        T_mean += p_not_found_yet

        p_found = 1.0 - p_not_found_yet * (1.0 - p_end)

        bin_to = bin_t
        bin_t = int((t - BIN_START_T) // BIN_SIZE_T)
        cdf_t[bin_to:bin_t] = cdf_t[bin_to]
        cdf_t[bin_t] = p_found

        bin_ho = bin_h
        bin_h = int((cumulative_hits - BIN_START_H) // BIN_SIZE_H)
        cdf_h[bin_ho:bin_h] = cdf_h[bin_ho]
        cdf_h[bin_h] = p_found

        p_not_found_yet *= 1 - p_end
        cumulative_hits += hit

        if p_not_found_yet < STOP_p or p_end > 1 - EPSILON or done:
            stop = 1
        elif t >= STOP_t - 1:
            stop = 2
            failed = 1
        elif myenv.agent_stuck:
            stop = 3
            failed = 1

        if stop:
            if episode % 100 == 0 and sys.stdout.isatty() and not WITH_MPI:
                txt = "episode: %7d [hit(t0) = %d]  ||  <nsteps>:  %7.2f  ||  max nb of steps:  %7.2f" \
                      % (episode, myenv.initial_hit, T_mean, t)
                if stop == 1:
                    print(txt)
                elif stop == 2:
                    print(txt + "  ||  max iter reached! (prob not found yet = %7.2e)" % p_not_found_yet)
                elif stop == 3:
                    print(txt + "  ||  agent stuck! (prob not found yet = %7.2e)" % p_not_found_yet)

                sys.stdout.flush()
            break

    # fill in remaining bins
    cdf_t[bin_t + 1:LEN_CDF_T] = cdf_t[bin_t]
    cdf_h[bin_h + 1:LEN_CDF_H] = cdf_h[bin_h]

    if np.any(cdf_t < 0) or np.any(cdf_h < 0):
        raise Exception("Missing values in the cdf, check implementation")

    # monitoring
    episode_elapsed_time = time.monotonic() - episode_start_time
    monitoring_episode_tmp_file = os.path.join(DIR_TMP, str("monitoring_episode_" + str(episode) + ".txt"))
    with open(monitoring_episode_tmp_file, "a") as mfile:
        mfile.write("%9d\t%4d\t%4d\t%10d\t%10.2e\t%.5e\n" % (
            episode, myenv.initial_hit, stop, near_boundaries, p_not_found_yet, episode_elapsed_time))

    return cdf_t, cdf_h, T_mean, failed


def run():
    """Main program that runs the episodes and computes the statistics.
    """
    if am_i_root():

        # Print parameters
        print("N_DIMS = " + str(N_DIMS))
        print("LAMBDA_OVER_DX = " + str(LAMBDA_OVER_DX))
        print("R_DT = " + str(R_DT))
        print("MU0_POISSON = " + str(MU0_POISSON))
        print("NORM_POISSON = " + NORM_POISSON)
        print("N_GRID = " + str(N_GRID))
        print("N_HITS = " + str(N_HITS))
        print("POLICY = " + str(POLICY))
        if POLICY == -1:
            print("MODEL_PATH = " + str(MODEL_PATH))
        else:
            print("STEPS_AHEAD = " + str(STEPS_AHEAD))
        print("EPSILON = " + str(EPSILON))
        print("STOP_t = " + str(STOP_t))
        print("STOP_p = " + str(STOP_p))
        print("N_PARALLEL = " + str(N_PARALLEL))
        print("WITH_MPI = " + str(WITH_MPI))
        print("ADAPTIVE_N_RUNS = " + str(ADAPTIVE_N_RUNS))
        print("REL_TOL = " + str(REL_TOL))
        print("MAX_N_RUNS = " + str(MAX_N_RUNS))
        print("N_RUNS(input) = " + str(N_RUNS))
        sys.stdout.flush()

    # Perform runs
    N_runs = N_RUNS
    if ADAPTIVE_N_RUNS or WITH_MPI:
        N_runs = int(N_PARALLEL * (np.ceil(N_runs / N_PARALLEL)))  # make it multiple of N_PARALLEL
        if am_i_root():
            print("N_RUNS(current) = " + str(N_runs))
            sys.stdout.flush()

    N_runso = 0

    if WITH_MPI:
        cdf_t_tot_loc = np.zeros(LEN_CDF_T, dtype=float)
        cdf_h_tot_loc = np.zeros(LEN_CDF_H, dtype=float)
        mean_t_loc = np.nan * np.ones(MAX_N_RUNS // N_PARALLEL, dtype=float)
        failed_loc = - np.ones(MAX_N_RUNS // N_PARALLEL, dtype=float)
    else:
        cdf_t_tot = np.zeros(LEN_CDF_T, dtype=float)
        cdf_h_tot = np.zeros(LEN_CDF_H, dtype=float)
        mean_t_episodes = np.nan * np.ones(MAX_N_RUNS, dtype=float)
        failed_episodes = - np.ones(MAX_N_RUNS, dtype=float)

    while True:
        if WITH_MPI:  # MPI
            if N_runs % N_PARALLEL != 0:
                raise Exception("N_runs must be multiple of N_PARALLEL with MPI")
            COMM.Barrier()
            # Decomposition
            Nepisodes = N_runs // N_PARALLEL
            episode_list = range(N_runso + ME, N_runs, N_PARALLEL)
            # Run episodes and reduce locally
            ind = N_runso // N_PARALLEL
            for episode in episode_list:
                cdf_t, cdf_h, mean_t_loc[ind], failed_loc[ind] = Worker(episode)
                cdf_t_tot_loc += cdf_t
                cdf_h_tot_loc += cdf_h
                ind += 1

            # Reduce globally the mean_t and failed
            mean_t_episodes = np.empty([N_runs], dtype=float)
            failed_episodes = np.empty([N_runs], dtype=float)
            COMM.Barrier()
            COMM.Allgather([mean_t_loc[:ind], Nepisodes, MPI.DOUBLE], [mean_t_episodes, Nepisodes, MPI.DOUBLE])
            COMM.Allgather([failed_loc[:ind], Nepisodes, MPI.DOUBLE], [failed_episodes, Nepisodes, MPI.DOUBLE])
            COMM.Barrier()
        elif N_PARALLEL > 1:  # multiprocessing
                # Run episodes in parallel
                pool = multiprocessing.Pool(N_PARALLEL)
                result = pool.map(Worker, range(N_runso, N_runs))
                pool.close()
                pool.join()
                # Reduce
                ind = N_runso
                for cdf_t, cdf_h, mean_t, failed in result:
                    cdf_t_tot += cdf_t
                    cdf_h_tot += cdf_h
                    mean_t_episodes[ind] = mean_t
                    failed_episodes[ind] = failed
                    ind += 1
        elif N_PARALLEL == 1:   # sequential
            # TODO check this gives the same results as in parallel
            pdfs_t = []
            ind = N_runso
            for episode in range(N_runso, N_runs):
                cdf_t, cdf_h, mean_t, failed = Worker(episode)
                cdf_t_tot += cdf_t
                cdf_h_tot += cdf_h
                mean_t_episodes[ind] = mean_t
                failed_episodes[ind] = failed
                ind += 1
        else:
            raise Exception("Problem with N_PARALLEL: must be an int >= 1")

        # estimate of the error
        mean_ep = np.mean(mean_t_episodes[:N_runs])
        sigma_ep = np.std(mean_t_episodes[:N_runs])
        std_error_mean = sigma_ep / np.sqrt(N_runs)
        rel_std_error_mean = std_error_mean / mean_ep

        # break clause
        if not ADAPTIVE_N_RUNS:
            break
        else:
            if rel_std_error_mean < REL_TOL:
                break
            elif N_runs >= MAX_N_RUNS:
                break
            else:
                N_runso = N_runs
                N_runs = int(np.ceil(1.05 * (sigma_ep / mean_ep / REL_TOL) ** 2))
                N_runs = min(N_runs, MAX_N_RUNS)
                N_runs = int(N_PARALLEL * (np.ceil(N_runs / N_PARALLEL)))  # make it multiple of N_PARALLEL
                if am_i_root():
                    print("N_RUNS(current) = " + str(N_runs))
                    sys.stdout.flush()

    # Reduce
    if WITH_MPI:
        # locally
        cdf_t_tot_loc /= N_runs
        cdf_h_tot_loc /= N_runs
        # Reduce globally
        cdf_t_tot = np.empty([LEN_CDF_T], dtype=float)
        cdf_h_tot = np.empty([LEN_CDF_H], dtype=float)
        COMM.Barrier()
        COMM.Allreduce(cdf_t_tot_loc, cdf_t_tot, op=MPI.SUM)
        COMM.Allreduce(cdf_h_tot_loc, cdf_h_tot, op=MPI.SUM)
        COMM.Barrier()
    else:
        cdf_t_tot /= N_runs
        cdf_h_tot /= N_runs
        mean_t_episodes = mean_t_episodes[:N_runs]
        failed_episodes = failed_episodes[:N_runs]

    # Further post processing, save and plot
    if am_i_root():
        print("N_RUNS(performed) = " + str(N_runs))
        sys.stdout.flush()

        # from cdf to pdf
        pdf_t_tot = cdf_to_pdf(cdf_t_tot)
        pdf_h_tot = cdf_to_pdf(cdf_h_tot)

        # compute stats of number of steps and number of hits
        t_bins = np.arange(BIN_START_T, BIN_END_T, BIN_SIZE_T) + 0.5 * BIN_SIZE_T
        mean_t, sigma_t, skew_t, kurt_t, p_found = stats_from_pdf(t_bins, pdf_t_tot)
        p25_t, p50_t, p75_t, p90_t, p95_t, p99_t, _ = stats_from_cdf(t_bins, cdf_t_tot)

        h_bins = np.arange(BIN_START_H, BIN_END_H, BIN_SIZE_H) + 0.5 * BIN_SIZE_H
        mean_h, sigma_h, skew_h, kurt_h, _ = stats_from_pdf(h_bins, pdf_h_tot)
        p25_h, p50_h, p75_h, p90_h, p95_h, p99_h, _ = stats_from_cdf(h_bins, cdf_h_tot)

        print("Number of steps: mean on %1d episodes  :  %.3f" % (N_runs, mean_t))
        print("Number of steps: median on %1d episodes:  %.3f" % (N_runs, p50_t))
        print("Number of steps: p99 on %1d episodes   :  %.3f" % (N_runs, p99_t))
        print("Number of steps: proba source not found:  %.10f" % (1 - p_found,))
        nb_failed = np.sum(failed_episodes)
        if np.any(failed_episodes < 0):
            nb_failed = -1
            print("Problem while recording failures")
        else:
            print("Number of failed episodes              :  %7d (%8.4f %%)" % (nb_failed, nb_failed / N_runs * 100))
        sys.stdout.flush()

        # save all parameters to txt file
        inputs = {
            "N_DIMS": N_DIMS,
            "LAMBDA_OVER_DX": LAMBDA_OVER_DX,
            "R_DT": R_DT,
            "MU0_POISSON": MU0_POISSON,
            "NORM_POISSON": NORM_POISSON,
            "N_GRID": N_GRID,
            "N_HITS": N_HITS,
            "POLICY": POLICY,
            "STEPS_AHEAD": STEPS_AHEAD,
            "MODEL_PATH": MODEL_PATH,
            "STOP_t": STOP_t,
            "STOP_p": STOP_p,
            "ADAPTIVE_N_RUNS": ADAPTIVE_N_RUNS,
            "REL_TOL": REL_TOL,
            "MAX_N_RUNS": MAX_N_RUNS,
            "N_RUNS_PERFORMED": N_runs,
            "BIN_START_T": BIN_START_T,
            "BIN_END_T": BIN_END_T,
            "BIN_SIZE_T": BIN_SIZE_T,
            "BIN_START_H": BIN_START_H,
            "BIN_END_H": BIN_END_H,
            "BIN_SIZE_H": BIN_SIZE_H,
            "EPSILON": EPSILON,
        }
        param_txt_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_parameters" + ".txt"))
        with open(param_txt_file, 'w') as out:
            for key, val in inputs.items():
                print(key + " = " + str(val), file=out)
        print(">>> Parameters saved in: " + param_txt_file)

        # save stats
        stats_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_statistics_failures" + ".txt"))
        with open(stats_file, "w") as sfile:
            sfile.write("p_not_found            \t%.8e\n" % (1 - p_found,))
            sfile.write("fraction_runs_stopnot1 \t%.8e\n" % (nb_failed / N_runs))
        print(">>> Statistics of failures saved in: " + stats_file)

        stats_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_statistics_nsteps" + ".txt"))
        with open(stats_file, "w") as sfile:
            for varname in (
            'mean_t', 'sigma_t', 'skew_t', 'kurt_t', 'p25_t', 'p50_t', 'p75_t', 'p90_t', 'p95_t', 'p99_t'):
                sfile.write("%s\t%.5e\n" % (varname, locals()[varname]))
        print(">>> Statistics of nsteps saved in: " + stats_file)

        stats_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_statistics_nhits" + ".txt"))
        with open(stats_file, "w") as sfile:
            for varname in (
            'mean_h', 'sigma_h', 'skew_h', 'kurt_h', 'p25_h', 'p50_h', 'p75_h', 'p90_h', 'p95_h', 'p99_h'):
                sfile.write("%s\t%.5e\n" % (varname, locals()[varname]))
        print(">>> Statistics of nhits saved in: " + stats_file)

        # save CDF of number of steps
        table_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_table_CDF_nsteps" + ".npy"))
        np.save(table_file, np.vstack((t_bins, cdf_t_tot)))
        print(">>> CDF(nsteps) saved in: " + table_file)

        # save CDF of number of hits
        table_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_table_CDF_nhits" + ".npy"))
        np.save(table_file, np.vstack((h_bins, cdf_h_tot)))
        print(">>> CDF(nhits) saved in: " + table_file)

        # save array of mean number of steps for each episode
        table_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_table_mean_nsteps_episodes" + ".npy"))
        np.save(table_file, mean_t_episodes)
        print(">>> mean(nsteps) for each episode saved in: " + table_file)

        # create and save figures
        if POLICY == -1:
            specifics = "MODEL = " + os.path.basename(MODEL_PATH)
        else:
            specifics = "STEPS_AHEAD = " + str(STEPS_AHEAD)
        subtitle = (
                "N_DIMS = "
                + str(N_DIMS)
                + ", "
                + "LAMBDA_OVER_DX = "
                + str(LAMBDA_OVER_DX)
                + ", "
                + "R_DT = "
                + str(R_DT)
                + ", "
                + "POLICY = "
                + str(POLICY)
                + ", "
                + specifics
                + ", "
                + "N_GRID = "
                + str(N_GRID)
                + ", "
                + "N_HITS = "
                + str(N_HITS)
                + ", "
                + "N_RUNS = "
                + str(N_runs)
                + "\n"
        )

        # plot PDF(nsteps), CDF(nsteps), PDF(nhits), CDF(nhits)
        for varname in ('number of steps', 'number of hits'):
            if varname == 'number of steps':
                bins = t_bins
                cdf_tot = cdf_t_tot
                pdf_tot = pdf_t_tot
                mean = mean_t
                sigma = sigma_t
                skew = skew_t
                kurt = kurt_t
                p50 = p50_t
                p75 = p75_t
                p90 = p90_t
                p99 = p99_t
                filesuffix = 'nsteps'
            else:
                bins = h_bins
                cdf_tot = cdf_h_tot
                pdf_tot = pdf_h_tot
                mean = mean_h
                sigma = sigma_h
                skew = skew_h
                kurt = kurt_h
                p50 = p50_h
                p75 = p75_h
                p90 = p90_h
                p99 = p99_h
                filesuffix = 'nhits'
            max_x = bins[np.nonzero(pdf_tot)[0][-1]]
            for fct in ("PDF", "CDF"):
                if fct == "PDF":
                    ydata = pdf_tot
                    ylim = (0.0, 1.02 * np.max(pdf_tot))
                elif fct == "CDF":
                    ydata = cdf_tot
                    ylim = (0.0, 1.0)
                fig1, ax1 = plt.subplots()
                ax1.plot(bins, ydata, "-o", markersize=2, linewidth=0.5)
                ax1.set_title(fct + " of " + varname)
                ax1.set_xlabel(varname + " to find the source")
                ax1.set_xlim((0, max_x + 1))
                ax1.set_ylim(ylim)
                plt.figtext(0.5, 0.995, subtitle, fontsize=3, ha="center", va="top")
                if fct == "PDF":
                    plt.figtext(0.90, 0.95, "mean = " + "{:.3e}".format(mean), fontsize=6, ha="right")
                    plt.figtext(0.90, 0.93, "std = " + "{:.3e}".format(sigma), fontsize=6, ha="right")
                    plt.figtext(0.90, 0.91, "skew = " + "{:.3e}".format(skew), fontsize=6, ha="right")
                    plt.figtext(0.90, 0.89, "ex. kurt = " + "{:.3e}".format(kurt), fontsize=6, ha="right")
                elif fct == "CDF":
                    plt.figtext(0.90, 0.95, "P50 = " + "{:.3e}".format(p50), fontsize=6, ha="right")
                    plt.figtext(0.90, 0.93, "P75 = " + "{:.3e}".format(p75), fontsize=6, ha="right")
                    plt.figtext(0.90, 0.91, "P90 = " + "{:.3e}".format(p90), fontsize=6, ha="right")
                    plt.figtext(0.90, 0.89, "P99 = " + "{:.3e}".format(p99), fontsize=6, ha="right")
                plt.grid(True)
                plt.draw()
                figure_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_figure_" + fct + "_" + filesuffix + ".pdf"))
                fig1.savefig(figure_file)
                plt.close(fig1)

                print(">>> Plot of " + fct + " of " + varname + " saved in: " + figure_file)

        # plot Hist of mean_episode(nsteps)
        mean_min = np.floor(np.min(mean_t_episodes))
        mean_max = np.ceil(np.max(mean_t_episodes))
        bins = np.arange(mean_min - 0.5, mean_max + 1.5)
        fig2, ax2 = plt.subplots()
        ax2.hist(mean_t_episodes, bins=bins, density=True)
        ax2.set_title("Histogram of mean number of steps")
        ax2.set_xlabel("mean number of steps")
        ax2.set_ylabel("frequency")
        plt.figtext(0.5, 0.995, subtitle, fontsize=3, ha="center", va="top")
        plt.figtext(0.90, 0.95, "mean = " + "{:.3e}".format(np.mean(mean_t_episodes)), fontsize=6, ha="right")
        plt.figtext(0.90, 0.93, "std = " + "{:.3e}".format(np.std(mean_t_episodes)), fontsize=6, ha="right")
        plt.figtext(0.90, 0.91, "skew = " + "{:.3e}".format(Skew(mean_t_episodes)), fontsize=6, ha="right")
        plt.figtext(0.90, 0.89, "ex. kurt = " + "{:.3e}".format(Kurt(mean_t_episodes)), fontsize=6, ha="right")
        plt.draw()
        figure_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_figure_PDF_mean_nsteps.pdf"))
        fig2.savefig(figure_file)
        plt.close(fig2)
        print(">>> Plot of Histogram(mean_episode(nsteps)) saved in: " + figure_file)

        # plot mean nb steps vs number of episodes
        number_episodes = range(1, N_runs + 1)
        cum_mean_t_episodes = np.cumsum(mean_t_episodes) / number_episodes
        if N_runs >= 100:
            number_episodes = number_episodes[20:]
            cum_mean_t_episodes = cum_mean_t_episodes[20:]
        fig3, ax3 = plt.subplots()
        ax3.plot(number_episodes, cum_mean_t_episodes)
        ax3.set_title("Convergence of the mean number of steps")
        ax3.set_xlabel("number of episodes")
        ax3.set_ylabel("mean number of steps")
        plt.figtext(0.5, 0.995, subtitle, fontsize=3, ha="center", va="top")
        plt.grid(True)
        plt.draw()
        figure_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_figure_convergence_mean_nsteps.pdf"))
        fig3.savefig(figure_file)
        plt.close(fig3)
        print(">>> Plot of convergence saved in: " + figure_file)

        # save monitoring information (concatenate episodes files)
        monitoring_episodes_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_monitoring_episodes.txt"))
        filenames = [os.path.join(DIR_TMP, str("monitoring_episode_" + str(episode) + ".txt")) for episode in range(N_runs)]
        with open(monitoring_episodes_file, "w") as mfile:
            mfile.write("# episode\tinit_hit\tstop\tboundaries\tp\t\telapsed(sec)\n")
            for fname in filenames:
                if os.path.isfile(fname):
                    with open(fname) as infile:
                        mfile.write(infile.read())
                    os.remove(fname)
                else:
                    print("Unexpected: Missing episode file: " + str(fname))

        # clean up tmp dirs
        if len(os.listdir(DIR_TMP)) != 0:
            print("Unexpected: The directory '" + DIR_TMP
                  + "' is not removed, because it should be empty but is not.")
        else:
            os.rmdir(DIR_TMP)
        if len(os.listdir(PARENT_DIR_TMP)) == 0:
            os.rmdir(PARENT_DIR_TMP)

        # summary
        monitoring_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_monitoring_summary" + ".txt"))
        with open(monitoring_file, "a") as mfile:
            mfile.write("*** initial hit ***\n")
            first_hit = np.loadtxt(monitoring_episodes_file, usecols=1, dtype='int')
            hit_max = np.max(first_hit)
            hit_hist, _ = np.histogram(first_hit, bins=np.arange(0.5, hit_max + 1.5), density=True)
            for h in range(1, hit_max + 1):
                mfile.write("hit=%1d: %6.2f %% \n" % (h, hit_hist[h - 1] * 100))

            mfile.write("\n*** stats convergence ***\n")
            mfile.write("number of runs (episodes) performed   : %7d\n" % N_runs)
            mfile.write("standard error of the mean (estimate) : %.5e = %5.2f %%\n"
                        % (std_error_mean, rel_std_error_mean * 100))

            stopping_reason = np.loadtxt(monitoring_episodes_file, usecols=2, dtype='int')
            stop_max = np.max(stopping_reason)
            stopping_hist, _ = np.histogram(stopping_reason, bins=np.arange(0.5, stop_max + 1.5), density=True)
            mfile.write("\n*** reason for stopping (1 is success, anything else is failure) ***\n")
            for stop in range(1, stop_max + 1):
                mfile.write("stop=%1d: %6.2f %% \n" % (stop, stopping_hist[stop - 1] * 100))

            mfile.write("\n*** proba source not found at the end of the episodes ***\n")
            p_not_found = np.loadtxt(monitoring_episodes_file, usecols=4)
            p_gtr_stop = p_not_found[p_not_found > STOP_p]
            p_not_found_max = np.max(p_not_found)
            mfile.write("STOP_p : %.5e\n" % STOP_p)
            mfile.write("max(p) : %.5e\n" % p_not_found_max)
            mfile.write("number of episodes where p > STOP_p : %7d (%8.4f %%)\n"
                        % (len(p_gtr_stop), len(p_gtr_stop) / N_runs * 100))

            near_boundaries = np.loadtxt(monitoring_episodes_file, usecols=3, dtype='int')
            near_boundaries = np.count_nonzero(near_boundaries)
            mfile.write("\n*** agent near boundaries ***\n")
            mfile.write("number of episodes where it happened : %7d (%8.4f %%)\n"
                        % (near_boundaries, near_boundaries / N_runs * 100))

            episode_elapsed = np.loadtxt(monitoring_episodes_file, usecols=5)
            mfile.write("\n*** computational cost per episode ***\n")
            mfile.write("avg elapsed hours per 'Worker(episode)' : %.5e\n" % (np.mean(episode_elapsed) / 3600.0))
            mfile.write("max elapsed hours per 'Worker(episode)' : %.5e\n" % (np.max(episode_elapsed) / 3600.0))

            elapsed_time_0 = (time.monotonic() - start_time_0) / 3600.0
            mfile.write("\n*** computational cost ***\n")
            mfile.write("total elapsed hours                             : %.5e\n" % elapsed_time_0)
            mfile.write("cost in hours = total elapsed time * N_PARALLEL : %.5e\n" % (elapsed_time_0 * N_PARALLEL))

        print(">>> Monitoring summary saved in: " + monitoring_file)

        sys.stdout.flush()


if __name__ == "__main__":

    # setup parallelization
    if WITH_MPI:
        COMM = MPI.COMM_WORLD
        ME = COMM.Get_rank()
        N_PARALLEL = COMM.Get_size()
        MASTER_PID = None
    else:
        COMM = None
        ME = None
        if N_PARALLEL <= 0:
            N_PARALLEL = os.cpu_count()
        MASTER_PID = os.getpid()

    # check and create directories
    if am_i_root():
        if not os.path.isdir(DIR_TMP):
            os.makedirs(DIR_TMP)
        elif len(os.listdir(DIR_TMP)) != 0:
            raise Exception("The directory '" + DIR_TMP + "' must be empty")
        if not os.path.isdir(DIR_OUTPUTS):
            os.makedirs(DIR_OUTPUTS)

    if WITH_MPI:
        COMM.Barrier()

    start_time_0 = time.monotonic()

    # autoset numerical parameters
    N_GRID, N_HITS, STOP_t, N_RUNS, MAX_N_RUNS, MU0_POISSON = autoset_numerical_parameters()
    if POLICY == -1:
        STEPS_AHEAD = None
    else:
        MODEL_PATH = None

    # define PDF bins
    BIN_START_T = -0.5
    BIN_SIZE_T = 1
    BIN_END_T = STOP_t + 1
    LEN_CDF_T = int(np.ceil((BIN_END_T - BIN_START_T) / BIN_SIZE_T))

    BIN_START_H = -0.5
    BIN_SIZE_H = 1
    if MU0_POISSON > 1:
        BIN_END_H = int(BIN_END_T * MU0_POISSON)
    else:
        BIN_END_H = int(BIN_END_T * np.sqrt(MU0_POISSON))
    LEN_CDF_H = int(np.ceil((BIN_END_H - BIN_START_H) / BIN_SIZE_H))

    # run
    if WITH_MPI:
        COMM.Barrier()
    run()
    if WITH_MPI:
        COMM.Barrier()

    if am_i_root():
        print("Completed. Time elapsed (in seconds):", time.monotonic() - start_time_0)
