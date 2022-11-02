import time
# Source-tracking POMDP
N_DIMS = 2  # number of space dimensions (1D, 2D, ...)
LAMBDA_OVER_DX = 1.0  # dimensionless problem size
R_DT = 2.0  # dimensionless source intensity
# Choose a policy
POLICY = 4  # O=infotaxis, 1=space-aware infotaxis, 2=custom, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
# Run name
RUN_NAME = "D" + str(N_DIMS) + "-L" + str(int(LAMBDA_OVER_DX)) + "-R" + str(int(R_DT)) + "_" + time.strftime("%Y%m%d-%H%M%S")