# ____________ FREE PARAMETERS ________________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 1
LAMBDA_OVER_DX = 2.0
R_DT = 2.0
# Policy
POLICY = 0  # -1: RL, O: infotaxis, 1: space-aware infotaxis, 5: random 6: greedy, 7: mean distance, 8: voting, 9: most likely state
MODEL_PATH = None  # only for POLICY=-1
STEPS_AHEAD = 1  # only for POLICY=0
# Setup
DRAW_SOURCE = True
ZERO_HIT = False
# Visualization
VISU_MODE = 2  # 0: run without video, 1: create video in the background, 2: create video and show live preview (slower)
FRAME_RATE = 5
KEEP_FRAMES = False
# ____________ DEFAULT PARAMETERS _____________________________________________________________________________________
# Source-tracking POMDP
NORM_POISSON = 'Euclidean'
N_HITS = None
N_GRID = None
# Stopping criteria
STOP_p = 1e-6
STOP_t = 1000000
# Saving
RUN_NAME = None



