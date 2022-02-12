import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'otto'))

import pytest
import numpy.testing as npt
import numpy as np
from copy import deepcopy
from classes.heuristicpolicies import HeuristicPolicies
from classes.training import TrainingEnv as env


def test_attributes():
    from scipy.special import kn
    N_DIMS = 2
    LAMBDA_OVER_DX = 2.0
    R_DT = 2.0
    INITIAL_HIT = 3
    myenv = env(
        Ndim=N_DIMS,
        lambda_over_dx=LAMBDA_OVER_DX,
        R_dt=R_DT,
        initial_hit=INITIAL_HIT
    )
    assert myenv.Ndim == N_DIMS
    assert myenv.lambda_over_dx == LAMBDA_OVER_DX
    assert myenv.R_dt == R_DT
    assert myenv.norm_Poisson == 'Euclidean'
    assert not myenv.draw_source
    assert myenv.Nactions == 2 * N_DIMS
    assert myenv.initial_hit == INITIAL_HIT
    assert not myenv.draw_source

    N_GRID = 37
    N_HITS = 4
    assert myenv.N == N_GRID
    assert myenv.Nhits == N_HITS

    assert myenv.mu0_Poisson == R_DT / np.log(2 * LAMBDA_OVER_DX) * kn(0, 1)

    hit_map = -np.ones([N_GRID] * N_DIMS, dtype=int)
    hit_map[N_GRID // 2, N_GRID // 2] = INITIAL_HIT
    npt.assert_array_equal(myenv.hit_map, hit_map)
    assert myenv.agent == [N_GRID // 2] * N_DIMS

    assert myenv.cumulative_hits == 0
    assert myenv.agent_near_boundaries == 0
    assert not myenv.agent_stuck
    assert myenv.obs == {"hit": INITIAL_HIT, "done": False}

    assert myenv._agento == [0] * N_DIMS
    assert myenv._agentoo == [N_GRID] * N_DIMS
    assert myenv._repeated_visits == 0

    assert hasattr(myenv, 'p_Poisson')


def test_sym_2d():
    myenv = env(
        Ndim=2,
        lambda_over_dx=1.0,
        R_dt=1.0,
    )
    myenv.p_source *= 0.0
    myenv.p_source[myenv.N // 2 - 1, myenv.N // 2] = 0.8
    myenv.p_source[myenv.N // 2, myenv.N // 2 - 2] = 0.2
    myenv.entropy = myenv._entropy(myenv.p_source)
    myenv.agent = [myenv.N // 2 - 3, myenv.N // 2 + 5]

    myenv.apply_sym_transformation(sym=1)
    assert myenv.agent == [myenv.N // 2 + 5, myenv.N // 2 - 3]
    assert myenv.p_source[myenv.N // 2 - 2, myenv.N // 2] == pytest.approx(0.2)
    assert myenv.p_source[myenv.N // 2, myenv.N // 2 - 1] == pytest.approx(0.8)

    myenv.apply_sym_transformation(sym=3)
    assert myenv.agent == [myenv.N // 2 + 5, myenv.N // 2 + 3]
    assert myenv.p_source[myenv.N // 2 - 2, myenv.N // 2] == pytest.approx(0.2)
    assert myenv.p_source[myenv.N // 2, myenv.N // 2 + 1] == pytest.approx(0.8)

    myenv.apply_sym_transformation(sym=3)
    myenv.apply_sym_transformation(sym=1)
    assert myenv.agent == [myenv.N // 2 - 3, myenv.N // 2 + 5]
    assert myenv.p_source[myenv.N // 2 - 1, myenv.N // 2] == pytest.approx(0.8)
    assert myenv.p_source[myenv.N // 2, myenv.N // 2 - 2] == pytest.approx(0.2)


def test_sym_1d():
    myenv = env(
        Ndim=1,
        lambda_over_dx=3.0,
        R_dt=1.0,
    )
    myenv.p_source *= 0.0
    myenv.p_source[myenv.N // 2 - 3] = 0.8
    myenv.entropy = myenv._entropy(myenv.p_source)
    myenv.agent = [myenv.N // 2 - 5]

    myenv.apply_sym_transformation(sym=1)
    assert myenv.agent == [myenv.N // 2 + 5]
    assert myenv.p_source[myenv.N // 2 + 3] == pytest.approx(0.8)
