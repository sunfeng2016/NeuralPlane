import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.models.JSBSim.envs.singlecontrol_env import SingleControlEnv

config_name="1/heading"

def main():
    env = SingleControlEnv(config_name)
    
if __name__ == "__main__":
    main()