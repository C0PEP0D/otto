import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'otto'))

import pytest
import numpy.testing as npt
import numpy as np
from classes.sourcetracking import SourceTracking as env
from classes.heuristicpolicy import HeuristicPolicy


def shared(params):
    assert params.N_DIMS == 1
    assert params.LAMBDA_OVER_DX == 2.0
    assert params.R_DT == 2.0
    assert params.NORM_POISSON == 'Euclidean'
    assert params.N_HITS is None
    assert params.N_GRID is None
    assert params.STOP_p == 1e-6
    assert params.RUN_NAME is None
    assert params.MODEL_PATH is None


def test_visualize_defaults():
    import visualize.parameters.__defaults as params
    shared(params)
    assert params.POLICY == 0
    assert params.STEPS_AHEAD == 1
    assert params.DRAW_SOURCE
    assert not params.ZERO_HIT
    assert params.VISU_MODE == 2
    assert params.FRAME_RATE == 5
    assert not params.KEEP_FRAMES
    assert params.STOP_t == 1000000


def test_evaluate_defaults():
    import evaluate.parameters.__defaults as params
    shared(params)
    assert params.POLICY == 0
    assert params.STEPS_AHEAD == 1
    assert params.N_RUNS is None
    assert params.ADAPTIVE_N_RUNS
    assert params.REL_TOL == 0.01
    assert params.MAX_N_RUNS == 100000
    assert params.N_PARALLEL == -1


def test_learn_defaults():
    import learn.parameters.__defaults as params
    shared(params)
    assert params.FC_LAYERS == 3
    assert params.FC_UNITS == 8
    assert params.MEMORY_SIZE == 1000
    assert params.E_GREEDY_DECAY == 1000
    assert params.ALGO_MAX_IT == 10000
    assert params.EVALUATE_PERFORMANCE_EVERY == 200
    assert params.N_PARALLEL == -1
    assert params.BATCH_SIZE == 64
    assert params.N_GD_STEPS == 12
    assert params.LEARNING_RATE == 0.001
    assert params.REPLAY_NTIMES == 4
    assert params.E_GREEDY_FLOOR == 0.1
    assert params.E_GREEDY_0 == 1.0
    assert params.SYM_EVAL_ENSEMBLE_AVG
    assert not params.SYM_TRAIN_ADD_DUPLICATES
    assert params.SYM_TRAIN_RANDOMIZE
    assert params.UPDATE_FROZEN_MODEL_EVERY == 1
    assert not params.DDQN
    assert params.POLICY_REF == 0
    assert params.N_RUNS_STATS is None
    assert params.PRINT_INFO_EVERY == 10
    assert params.SAVE_MODEL_EVERY == 10
    assert params.STOP_t is None
