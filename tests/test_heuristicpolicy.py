import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'otto'))

import pytest
import numpy.testing as npt
import numpy as np
from classes.sourcetracking import SourceTracking as env
from classes.heuristicpolicy import HeuristicPolicy


def test_attributes():
    myenv = env(
        Ndim=1,
        lambda_over_dx=1.0,
        R_dt=1.0,
    )

    POLICY = 6
    mypol = HeuristicPolicy(
        env=myenv,
        policy=POLICY,
    )

    assert mypol.policy == POLICY
    assert mypol.steps_ahead == 1
    assert mypol.discount is None

    POLICY = 0
    STEPS_AHEAD = 2
    mypol = HeuristicPolicy(
        env=myenv,
        policy=POLICY,
        steps_ahead=STEPS_AHEAD,
    )
    assert mypol.env == myenv
    assert mypol.policy == POLICY
    assert mypol.steps_ahead == STEPS_AHEAD
    assert mypol.discount == 0.999

    DISCOUNT = 0.9
    mypol = HeuristicPolicy(
        env=myenv,
        policy=POLICY,
        steps_ahead=STEPS_AHEAD,
        discount=DISCOUNT
    )
    assert mypol.discount == DISCOUNT


def setup_noisy():
    myenv = env(
        Ndim=2,
        lambda_over_dx=1.0,
        R_dt=1.0,
    )
    myenv.p_source *= 0.0
    myenv.p_source += 1 / myenv.N ** myenv.Ndim
    myenv.p_source[myenv.N // 2 + 1, myenv.N // 2] = 0.2
    myenv.p_source[myenv.N // 2, myenv.N // 2 + 1] = 0.8
    myenv.p_source /= np.sum(myenv.p_source)
    myenv.entropy = myenv._entropy(myenv.p_source)
    return myenv


def setup_clear():
    myenv = env(
        Ndim=2,
        lambda_over_dx=1.0,
        R_dt=1.0,
    )
    myenv.p_source *= 0.0
    myenv.p_source[myenv.N // 2 + 1, myenv.N // 2] = 0.5
    myenv.p_source[myenv.N // 2, myenv.N // 2 + 1] = 0.5
    myenv.entropy = myenv._entropy(myenv.p_source)
    return myenv


def test_behavior():
    myenv = setup_noisy()
    for policy in (0, 1, 6, 7, 8, 9):
        mypol = HeuristicPolicy(
            env=myenv,
            policy=policy,
        )
        action = mypol.choose_action()
        assert action == 3


def test_policies_1():
    myenv = setup_noisy()
    mypol = HeuristicPolicy(
        env=myenv,
        policy=0,
    )
    # greedy
    _, out = mypol._greedy_policy()
    assert out[0] == out[2]
    assert out[3] == pytest.approx(1.0 - myenv.p_source[myenv.N // 2, myenv.N // 2 + 1])

    # voting
    _, out = mypol._voting_policy()
    assert out[0] == out[2]
    assert out[3] > out[1] > out[2]

    # most likely state
    _, out = mypol._most_likely_state_policy()
    assert out[0] == out[1] == out[2]
    assert out[3] == pytest.approx(0.0)


def test_policies_2():
    myenv = setup_clear()
    mypol = HeuristicPolicy(
        env=myenv,
        policy=0,
    )
    # infotaxis
    _, out = mypol._infotaxis()
    assert out[0] == out[2]
    assert out[1] == out[3] == pytest.approx(myenv.entropy)

    # space-aware infotaxis
    _, out = mypol._space_aware_infotaxis()
    assert out[0] == out[2]
    assert out[1] == out[3] == pytest.approx(0.5)

    # mean distance
    _, out = mypol._mean_distance_policy()
    assert out[0] == out[2] == pytest.approx(2.0)
    assert out[1] == out[3] == pytest.approx(1.0)


def test_nstep_infotaxis():
    STEPS_AHEAD = 3
    myenv = env(
        Ndim=2,
        lambda_over_dx=1.0,
        R_dt=1.0,
    )
    myenv.p_source[myenv.N // 2, myenv.N // 2 + 2] += 0.1
    myenv.p_source[myenv.N // 2, myenv.N // 2 + 3] += 0.5
    myenv.p_source[myenv.N // 2, myenv.N // 2 + 4] += 1.0
    myenv.p_source /= np.sum(myenv.p_source)
    myenv.entropy = myenv._entropy(myenv.p_source)
    mypol = HeuristicPolicy(
            env=myenv,
            policy=0,
            steps_ahead=STEPS_AHEAD,
    )
    action = mypol.choose_action()
    a1, out1 = mypol._infotaxis_n_steps_no_discount(steps_ahead=STEPS_AHEAD)
    a2, out2 = mypol._infotaxis_n_steps(steps_ahead=STEPS_AHEAD, discount=1.0)

    assert action == a1 == a2
    assert action == 3
    npt.assert_allclose(out1, out2)

    a0, out0 = mypol._infotaxis()
    a3, out3 = mypol._infotaxis_n_steps(steps_ahead=STEPS_AHEAD, discount=0.0)

    assert a0 == a3
    npt.assert_allclose(out0, out3)


