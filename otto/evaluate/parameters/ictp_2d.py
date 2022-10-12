# Source-tracking POMDP (do not change)
N_DIMS = 2  # number of space dimensions (1D, 2D, ...)
LAMBDA_OVER_DX = 1.0  # dimensionless problem size
R_DT = 2.0  # dimensionless source intensity
# Choose a policy
POLICY = 0  # O=infotaxis, 1=space-aware infotaxis, 2=custom, 5=random, 6=greedy, 7=mean distance, 8=voting, 9=mls
# Control how many episodes are run: tolerance on the relative error on the mean number of steps to find the source
REL_TOL = 0.01  # (0.01 is default, use 0.02 for a faster evaluation)
