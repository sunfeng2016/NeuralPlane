import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class StabilityReward(BaseRewardFunction):
    """
    Measure the difference between the current posture and the target posture
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        P, Q, R = env.model.get_angular_velocity()
        u_norm = env.model.get_control_norm()
        
        # reward_p = -0.5 * P ** 2
        # reward_q = -0.1 * Q ** 2
        # reward_r = -0.1 * R ** 2
        # reward_u = -0.01 * u_norm ** 2
        # reward_stability = reward_p + reward_q + reward_r + reward_u
        
        reward_p = torch.exp(-0.5 * P ** 2)
        reward_q = torch.exp(-0.1 * Q ** 2)
        reward_r = torch.exp(-0.1 * R ** 2)
        reward_u = torch.exp(-0.01 * u_norm ** 2)
        reward_stability = (reward_p * reward_q * reward_r * reward_u) ** (1/4)
        
        return reward_stability
