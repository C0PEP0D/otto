
# Source-tracking POMDP
N_DIMS = 2
LAMBDA_OVER_DX = 2.0
R_DT = 2.0
# Neural network architecture
FC_LAYERS = 3
FC_UNITS = 512
# Experience replay
MEMORY_SIZE = 1000
# Exploration: eps is the probability of taking a random action when executing the policy
E_GREEDY_DECAY = 10000   # eps = E_GREEDY_0 * exp(-i/E_GREEDY_DECAY), where i is the algo iteration
# Max number of training iterations
ALGO_MAX_IT = 1000000
# Evaluation of the RL policy
EVALUATE_PERFORMANCE_EVERY = 1000
# Restart from saved model
MODEL_PATH = None  # e.g. "./models/20220201-230054/20220201-230054_value_model"
# Parallelization: how many episodes are computed in parallel (how many cores are used)
N_PARALLEL = 1    # -1 for using all cores, 1 for sequential (useful as parallel code may hang with larger NN)
