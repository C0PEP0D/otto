import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..', 'otto'))
import pytest
import numpy.testing as npt
import numpy as np
from copy import deepcopy
from classes.rlpolicy import RLPolicy
from classes.training import TrainingEnv as env
from classes.valuemodel import ValueModel, reload_model

MODEL_PATH = os.path.join(sys.path[0], '..', 'zoo', 'models', 'zoo_model_2_1_4')


def test_choose_action():
    for sym_avg in (False, True):
        myenv = env(
            Ndim=2,
            lambda_over_dx=1.0,
            R_dt=4.0,
            initial_hit=1,
        )
        mymodel = reload_model(
            MODEL_PATH,
            inputshape=myenv.NN_input_shape
        )
        mypol = RLPolicy(
            env=myenv,
            model=mymodel,
            sym_avg=sym_avg,
        )

        hit, p_end, done = myenv.step(0, hit=0)
        hit, p_end, done = myenv.step(1, hit=0)
        hit, p_end, done = myenv.step(1, hit=1)

        action_ref = mypol.choose_action()
        action0, value0 = mypol._value_policy()

        state, statep = myenv.transitions()
        action1, value1 = myenv._value_policy_from_statep(model=mymodel, statep=statep, sym_avg=sym_avg)

        assert action_ref == action0 == action1
        npt.assert_array_equal(value0, value1)


