import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'otto'))

import pytest
import numpy.testing as npt
import numpy as np
from classes.sourcetracking import SourceTracking as env


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


def test_initial_p_source():
    myenv = env(
        Ndim=1,
        lambda_over_dx=4.0,
        R_dt=3.0,
        initial_hit=1,
    )

    # analytical distance
    dist = np.arange(myenv.N, dtype=int)
    dist = np.abs(dist - myenv.N // 2)

    # test distance function
    distance = myenv._distance()
    npt.assert_array_equal(distance, dist)

    # analytical expression of p_source after hit = 1 in 1D
    m = myenv.mu0_Poisson * np.exp(1)
    p_source = m * np.exp(-distance/myenv.lambda_over_dx - m * np.exp(-distance/myenv.lambda_over_dx))
    p_source = 2 * p_source / (1.0 - np.exp(- m * np.exp(- 0.5 / myenv.lambda_over_dx)))
    p_source[myenv.N // 2] = 0.0
    p_source = p_source / np.sum(p_source)

    # test initial p_source
    npt.assert_allclose(myenv.p_source, p_source)

    # test the update function
    myenv.p_source = np.ones([myenv.N]) / (myenv.N - 1)
    myenv._update_p_source(hit=1)
    npt.assert_allclose(myenv.p_source, p_source)


def test_misc_methods():

    for Ndim in (1, 2, 3):
        myenv = env(
            Ndim=Ndim,
            lambda_over_dx=1.0,
            R_dt=1.0,
        )
        uniform = np.ones([myenv.N] * myenv.Ndim) / (myenv.N ** myenv.Ndim)
        # test entropy
        assert np.abs(myenv._entropy(uniform) - myenv.Ndim * np.log2(myenv.N)) < 1e-10

        # test near boundaries
        assert not myenv._is_agent_near_boundaries(n_boundary=1)
        myenv.agent[0] = 0
        assert myenv._is_agent_near_boundaries(n_boundary=1)

        # test move
        new_agent, is_move_possible = myenv._move(1, myenv.agent)
        assert is_move_possible
        new_agent, _ = myenv._move(1, new_agent)
        assert new_agent[0] == 2


def test_step():
    myenv = env(
        Ndim=2,
        lambda_over_dx=1.0,
        R_dt=1.0,
    )
    myenv.p_source[myenv.N // 2 - 1, myenv.N // 2] += 0.8
    myenv.p_source = myenv.p_source / np.sum(myenv.p_source)
    myenv.entropy = myenv._entropy(myenv.p_source)
    p_end_check = myenv.p_source[myenv.N // 2 - 1, myenv.N // 2]

    hit, p_end, done = myenv.step(action=0, hit=2)

    assert hit == 2
    assert p_end == pytest.approx(p_end_check)
    assert not done
