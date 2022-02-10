# ____________ FREE PARAMETERS ________________________________________________________________________________________
# Source-tracking POMDP
N_DIMS = 1
LAMBDA_OVER_DX = 2.0
R_DT = 2.0
# Neural network (NN) architecture
FC_LAYERS = 3
FC_UNITS = 8
# Experience replay
MEMORY_SIZE = 1000
# Exploration: eps is the probability of taking a random action when executing the policy
E_GREEDY_DECAY = 1000   # eps = E_GREEDY_0 * exp(-i/E_GREEDY_DECAY), where i is the algo iteration
# Max number of training iterations
ALGO_MAX_IT = 10000
# Evaluation of the RL policy
EVALUATE_PERFORMANCE_EVERY = 200
# Restart from saved model
MODEL_PATH = None  # e.g. "./models/20220201-230054/20220201-230054_value_model"
# Parallelization: how many episodes are computed in parallel (how many cores are used)
N_PARALLEL = -1    # -1 for using all cores, 1 for sequential (useful as parallel code may hang with larger NN)

# ____________ DEFAULT RL PARAMETERS __________________________________________________________________________________
# Stochastic gradient descent
BATCH_SIZE = 64
N_GD_STEPS = 12
LEARNING_RATE = 0.001
# Experience replay
REPLAY_NTIMES = 4
# Exploration: eps is the probability of taking a random action when executing the policy
E_GREEDY_FLOOR = 0.1  # floor value of eps (cannot be smaller than E_GREEDY_FLOOR)
E_GREEDY_0 = 1.0
# Accounting for symmetries
SYM_EVAL_ENSEMBLE_AVG = True
SYM_TRAIN_ADD_DUPLICATES = False
SYM_TRAIN_RANDOMIZE = True
# Additional DQN algo parameters
UPDATE_FROZEN_MODEL_EVERY = 1
DDQN = False
# Evaluation of the RL policy
POLICY_REF = 0
N_RUNS_STATS = None
# Monitoring/Saving during the training
PRINT_INFO_EVERY = 10
SAVE_MODEL_EVERY = 10

# ____________ DEFAULT OTHER PARAMETERS _______________________________________________________________________________
# Source-tracking POMDP
NORM_POISSON = 'Euclidean'
N_HITS = None
N_GRID = None
# Criteria for terminating an episode
STOP_t = None
STOP_p = 1E-6
# Saving
RUN_NAME = None



