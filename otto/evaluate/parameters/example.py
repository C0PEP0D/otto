
# Source-tracking POMDP
N_DIMS = 2
LAMBDA_OVER_DX = 2.0
R_DT = 2.0
# Policy
POLICY = 0  # -1: RL, O: infotaxis, 1: space-aware infotaxis, 5: random, 6: greedy, 7: mean distance, 8: voting, 9: most likely state
MODEL_PATH = None  # only for POLICY=-1, e.g. "../train/models/20220201-230054/20220201-230054_value_model"
STEPS_AHEAD = 1  # only for POLICY=0

