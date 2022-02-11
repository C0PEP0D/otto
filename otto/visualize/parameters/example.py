
# Source-tracking POMDP
N_DIMS = 2
LAMBDA_OVER_DX = 2.0
R_DT = 2.0
# Policy
POLICY = 0  # -1: RL, O: infotaxis, 1: space-aware infotaxis, 5: random, 6: greedy, 7: mean distance, 8: voting, 9: most likely state
MODEL_PATH = None  # only for POLICY=-1, e.g. "../learn/models/20220201-230054/20220201-230054_value_model"
STEPS_AHEAD = 1  # only for POLICY=0
# Setup
DRAW_SOURCE = True
ZERO_HIT = False
# Visualization
VISU_MODE = 2  # 0: run without video, 1: create video in the background, 2: create video and show live preview (slower)
FRAME_RATE = 5
KEEP_FRAMES = False
