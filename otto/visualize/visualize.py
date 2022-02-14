#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to visualize a search in 1D, 2D or 3D.
Frame generation and live preview work on all platforms, but video recording is Linux-only and requires ffmpeg.

The list of all parameters is given below.
Default parameters are set by '__default.py' in the local 'parameters' directory.

Source-tracking POMDP
    - N_DIMS (int > 0)
        number of dimension (1D, 2D, ...)
    - LAMBDA_OVER_DX (float >= 1)
        sets the dimensionless problem size (odor dispersion lengthscale divided by agent's step size)
    - R_DT (float > 0)
        sets the dimensionless source intensity (source rate of emission multiplied by the agent's time step)
    - NORM_POISSON ('Euclidean', 'Manhattan' or 'Chebyshev')
        norm used for hit detections
    - N_HITS (int >= 2 or None)
        number of values of hit possible, set automatically if None
    - N_GRID (int >=3 or None)
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
    - STEPS_AHEAD (int>=1)
        number of anticipated future moves, only for POLICY=0

Setup
    - DRAW_SOURCE (bool)
        whether to actually draw the source location (otherwise uses Bayesian framework)
    - ZERO_HIT (bool)
        whether to force a series of zero hits

Visualization
    - VISU_MODE (int={0,1,2})
        - 0: run without video
        - 1: make video in the background
        - 2: make video with live preview (slower)
    - FRAME_RATE (int)
        number of frames per second in the video
    - KEEP_FRAMES (bool)
        whether each frame is conserved (otherwise frames are deleted, only the video is kept)

Stopping criteria (only if DRAW_SOURCE is False)
    - STOP_p (float ~ 0):
        stops when the probability that the source is found is greater than 1 - STOP_p
    - STOP_t (int)
        max number of iterations

Saving
    - RUN_NAME (str or None)
        prefix used for all output files, if None will use timestamp
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], '..', '..')))
sys.path.insert(2, os.path.abspath(os.path.join(sys.path[0], '..', '..', 'zoo')))
import time
import argparse
import importlib
from otto.classes.sourcetracking import SourceTracking as env
from otto.classes.visualization import Visualization

# import default globals
from otto.visualize.parameters.__defaults import *

# import globals from user defined parameter file
if os.path.basename(sys.argv[0]) not in ["sphinx-build", "build.py"]:
    parser = argparse.ArgumentParser(description='Visualize an episode')
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

DIR_OUTPUTS = os.path.abspath(os.path.join(sys.path[0], "outputs"))

# _______________________________________________


def run():
    """Main function"""
    print("*** initializing env...")

    myenv = env(
        Ndim=N_DIMS,
        lambda_over_dx=LAMBDA_OVER_DX,
        R_dt=R_DT,
        norm_Poisson=NORM_POISSON,
        Ngrid=N_GRID,
        Nhits=N_HITS,
        draw_source=DRAW_SOURCE,
    )
    print("N_DIMS = " + str(myenv.Ndim))
    print("LAMBDA_OVER_DX = " + str(myenv.lambda_over_dx))
    print("R_DT = " + str(myenv.R_dt))
    print("NORM_POISSON = " + myenv.norm_Poisson)
    print("N_GRID = " + str(myenv.N))
    print("N_HITS = " + str(myenv.Nhits))

    if POLICY == -1:
        mymodel = reload_model(MODEL_PATH, inputshape=myenv._inputshape())
        mypol = RLPolicy(
            env=myenv,
            model=mymodel,
        )
        print("POLICY = -1 (" + mypol.policy_name + ")")
        print("MODEL_PATH =", MODEL_PATH)
        print("MODEL_CONFIG =", mymodel.config)
    else:
        mypol = HeuristicPolicy(
            env=myenv,
            policy=POLICY,
            steps_ahead=STEPS_AHEAD,
        )
        print("POLICY = " + str(mypol.policy_index) + " (" + mypol.policy_name + ")")
        print("STEPS_AHEAD = " + str(mypol.steps_ahead))

    print("*** running...")

    t = 0  # timestep
    T_mean = 0
    p_not_found_yet = 1   # proba the source has not been found yet
    stop = 0

    if VISU_MODE > 0:
        filename = os.path.join(DIR_OUTPUTS, str(RUN_NAME))
        myvisu = Visualization(
            myenv,
            live=VISU_MODE == 2,
            filename=filename
        )

    while True:

        if VISU_MODE > 0:
            if DRAW_SOURCE:
                toptext = "current policy: %s, current step: %d" % (mypol.policy_name, t)
            else:
                toptext = "current policy: %s, current step: %d, proba not found: %.3f %%" \
                          % (mypol.policy_name, t, p_not_found_yet * 100)
            myvisu.record_snapshot(num=t, toptext=toptext)

        # choice of action
        action = mypol.choose_action()

        # step in myenv
        forced_hit = None
        if ZERO_HIT:
            forced_hit = 0
        hit, p_end, done = myenv.step(action, hit=forced_hit)

        t += 1
        T_mean += p_not_found_yet
        p_not_found_yet *= 1 - p_end

        print("nstep: %4d, action: %1d, hits: %3d, cum_hits: %6d, p_not_found_yet: %f"
              % (t, action, hit, myenv.cumulative_hits, p_not_found_yet))

        if done and DRAW_SOURCE:
            message = "stopped because: source found"
            stop = 1
        elif p_not_found_yet < STOP_p or p_end > 1 - EPSILON:
            message = "stopped because: source (almost surely) found " \
                      "(p_not_found_yet = " + str(p_not_found_yet) + ")"
            stop = 1
        elif t >= STOP_t - 1:
            message = "stopped because failure: max number of iterations reached (nb it = " + str(c) + ")"
            stop = 2
        elif myenv.agent_stuck:
            message = "stopped because failure: agent is stuck"
            stop = 3

        if stop:
            break

    print("*** complete")
    print(message)
    if not DRAW_SOURCE:
        print("mean number of steps on this episode =", T_mean)

    if VISU_MODE > 0:
        if DRAW_SOURCE:
            toptext = "current policy: %s, current step: %d" % (mypol.policy_name, t)
        else:
            toptext = "current policy: %s, current step: %d, proba not found: %.3f %%" % (mypol.policy_name, t, p * 100)

        myvisu.record_snapshot(num=t, toptext=toptext)
        exitcode = myvisu.make_video(frame_rate=FRAME_RATE, keep_frames=KEEP_FRAMES)
        if exitcode == 0:
            print(">>> Video saved in: " + str(filename) + "_video.mp4")
        else:
            print(">>> Frames have been saved in: " + str(filename) + "_frames")


if __name__ == "__main__":

    if not os.path.isdir(DIR_OUTPUTS):
        os.makedirs(DIR_OUTPUTS)

    start_time_0 = time.monotonic()
    run()
    if VISU_MODE == 0:
        print("CPU time (in seconds):", time.monotonic() - start_time_0)
