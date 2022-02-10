import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'otto'))

import pytest
import numpy.testing as npt
import numpy as np
from classes.sourcetracking import SourceTracking as env
from classes.visualization import Visualization


def test_env_attributes():
    N_DIMS = 1
    LAMBDA_OVER_DX = 2.0
    R_DT = 2.0
    POLICY = 0
    STEPS_AHEAD = 2

    myenv = env(
        Ndim=N_DIMS,
        lambda_over_dx=LAMBDA_OVER_DX,
        R_dt=R_DT,
    )
    mypol = HeuristicPolicies(
        env=myenv,
        policy=POLICY,
        steps_ahead=STEPS_AHEAD,
    )

    assert myenv.Ndim == N_DIMS
    assert myenv.lambda_over_dx == LAMBDA_OVER_DX
    assert myenv.R_dt == R_DT
    assert myenv.norm_Poisson == 'Euclidean'
    assert not myenv.draw_source
    assert myenv.Nactions == 2 * N_DIMS
    assert mypol.env == env

    assert mypol.policy == POLICY
    assert mypol.steps_ahead == STEPS_AHEAD
    assert mypol.discount == 0.999

