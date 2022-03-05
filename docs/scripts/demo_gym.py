"""How to use the gym environment for the source-tracking POMDP."""
import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], '..', '..')))
from otto.classes.sourcetracking import SourceTracking
from otto.classes.gymwrapper import GymWrapper


def run():

    mypomdp = SourceTracking(
        Ndim=1,
        lambda_over_dx=1.0,
        R_dt=2,
        draw_source=True,
    )

    myenv = GymWrapper(sim=mypomdp)

    # Execute random policy
    done = False
    cum_reward = 0.0
    observation = myenv.reset()
    print("---- t=0, observation=" + str(observation))
    while not done:
        action = myenv.action_space.sample()
        observation, reward, done, info = myenv.step(action)
        cum_reward += reward
        print(
            "---- t=" + str(myenv.t)
            + ", action=" + str(action)
            + ", hit=" + str(info["hit"])
            + ", reward=" + str(reward)
            + ", return=" + str(cum_reward)
            + ", timeout=" + str(info["timeout"])
            + ", found= " + str(info["found"])
            + ", observation=" + str(observation)
            )

    print("Done. Return=" + str(cum_reward))


if __name__ == "__main__":

    run()

