# ____________ BASIC PARAMETERS _______________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 3  # number of space dimensions (1D, 2D, ...)
LAMBDA_OVER_DX = 1.0  # dimensionless problem size
R_DT = 2.0  # dimensionless source intensity
# Neural network (NN) architecture
FC_LAYERS = 3  # number of hidden layers
FC_UNITS = 512  # number of units per layers
# Experience replay
MEMORY_SIZE = 1000  # number of transitions (s, s') to keep in memory
# Exploration: eps is the probability of taking a random action when executing the policy
E_GREEDY_DECAY = 10000   # timescale for eps decay, in number of training iterations
# Max number of training iterations
ALGO_MAX_IT = 100000  # max number of training iterations
# Evaluation of the RL policy
EVALUATE_PERFORMANCE_EVERY = 10000  # how often is the RL policy evaluated, in number of training iterations
# Restart from saved model, if None start from scratch
MODEL_PATH = None  # path to saved model, e.g., "./models/20220201-230054/20220201-230054_model"
# Parallelization: how many episodes are computed in parallel (how many cores are used)
N_PARALLEL = 1    # -1 for using all cores, 1 for sequential (useful as parallel code may hang with larger NN)
# Numer of episodes for evaluation
N_RUNS_STATS = 1000

