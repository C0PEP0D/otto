# Source-tracking POMDP
N_DIMS = 2  # number of space dimensions (1D, 2D, ...)
LAMBDA_OVER_DX = 1.0  # dimensionless problem size
R_DT = 2.0  # dimensionless source intensity
# Choose a policy
POLICY = 4  # O=infotaxis, 1=space-aware infotaxis, 2=custom, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
# Parameters for the video
DRAW_SOURCE = False
VISU_MODE = 1  # 0: run without video (very fast but nothing is saved), 1: create video in the background, 2: create video and show live preview (slower)
FRAME_RATE = 5  # number of frames per second in the video
KEEP_FRAMES = False  # whether individual frames should be saved (otherwise frames are deleted, only the video is kept)