# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 1  # number of space dimensions (1D, 2D, ...)
LAMBDA_OVER_DX = 2.0  # dimensionless problem size
R_DT = 2.0  # dimensionless source intensity
# Policy
POLICY = 0  # -1=RL, O=infotaxis, 1=space-aware infotaxis, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
MODEL_PATH = None  # saved model for POLICY=-1, e.g., "../learn/models/20220201-230054/20220201-230054_value_model"
STEPS_AHEAD = 1  # number of anticipated moves, can be > 1 only for POLICY=0
# Parallelization
N_PARALLEL = -1  # -1 for using all cores, 1 for sequential (useful as parallel code may hang with larger NN)
