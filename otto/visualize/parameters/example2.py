# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 2  # number of space dimensions (1D, 2D, ...)
LAMBDA_OVER_DX = 1.0  # dimensionless problem size
R_DT = 2.0  # dimensionless source intensity
# Policy
POLICY = 1  # -1=RL, O=infotaxis, 1=space-aware infotaxis, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
MODEL_PATH = None  # saved model for POLICY=-1, e.g., "../learn/models/20220201-230054/20220201-230054_model"
STEPS_AHEAD = 1  # number of anticipated moves, can be > 1 only for POLICY=0
# Setup
DRAW_SOURCE = False  # if False, episodes will continue until the source is almost surely found (Bayesian setting)
ZERO_HIT = False  # whether to enforce a series of zero hits
# Visualization
VISU_MODE = 2  # 0: run without video, 1: create video in the background, 2: create video and show live preview (slower)
FRAME_RATE = 5  # number of frames per second in the video
KEEP_FRAMES = False  # whether individual frames should be saved (otherwise frames are deleted, only the video is kept)
