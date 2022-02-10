# ____________ FREE PARAMETERS ________________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 1
LAMBDA_OVER_DX = 2.0
R_DT = 2.0
# Policy
POLICY = 0  # -1: RL, O: infotaxis, 1: space-aware infotaxis, 5: random, 6: greedy, 7: mean distance, 8: voting, 9: most likely state
MODEL_PATH = None  # only for POLICY=-1
STEPS_AHEAD = 1  # only for POLICY=0

# ____________ DEFAULT PARAMETERS _____________________________________________________________________________________
# Source-tracking POMDP
NORM_POISSON = 'Euclidean'
N_HITS = None
N_GRID = None
# Statistics computation
N_RUNS = None
ADAPTIVE_N_RUNS = True
REL_TOL = 0.01
MAX_N_RUNS = 100000
STOP_p = 1e-6
# Saving
RUN_NAME = None
# Parallelization
N_PARALLEL = -1


