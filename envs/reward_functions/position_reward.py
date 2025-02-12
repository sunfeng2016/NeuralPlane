import os
import sys
import math
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class PositionReward(BaseRewardFunction):
    """
    Measure the difference between the current position and the target position
    """
    def __init__(self, config):
        super().__init__(config)
        self.error_scale = getattr(config, 'error_scale', 50.0)
        self.max_reward = getattr(config, 'max_reward', 1.0)
        
    def get_reward_0122(self, task, env): # 0122
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        # 获取当前智能体位置和目标位置
        npos, epos, altitude = env.model.get_position()
        target_npos, target_epos, target_altitude = task.target_npos, task.target_epos, task.target_altitude
        
        new_distances = torch.sqrt(
            (npos - target_npos) ** 2 +
            (epos - target_epos) ** 2 +
            (altitude - target_altitude) ** 2
        )
        
        reward = (task.min_distance2target - new_distances) * 0.01
        task.min_distance2target = torch.where(
            new_distances < task.min_distance2target,
            new_distances,
            task.min_distance2target
        )

        # print("*" * 20)
        # print(reward)
        # print(torch.mean(reward))
        
        return reward
        
    def get_reward(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        # 获取当前智能体位置和目标位置
        npos, epos, altitude = env.model.get_position()
        target_npos, target_epos, target_altitude = task.target_npos, task.target_epos, task.target_altitude

        # 计算当前位置与目标位置的误差
        prev_delta_npos = torch.abs(task.cur_npos - target_npos) * 0.3048  # 单位转换为米
        prev_delta_epos = torch.abs(task.cur_epos - target_epos) * 0.3048
        prev_delta_altitude = torch.abs(task.cur_altitude - target_altitude) * 0.3048

        curr_delta_npos = torch.abs(npos - target_npos) * 0.3048
        curr_delta_epos = torch.abs(epos - target_epos) * 0.3048
        curr_delta_altitude = torch.abs(altitude - target_altitude) * 0.3048
        
        reward_npos = prev_delta_npos - curr_delta_npos
        reward_epos = prev_delta_epos - curr_delta_epos
        reward_altitude = prev_delta_altitude - curr_delta_altitude
        
        reward = (reward_npos + reward_epos + reward_altitude) * 0.1

        return reward
        
    def get_reward_0113(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        npos, epos, altitude = env.model.get_position()
        target_npos, target_epos, target_altitude = task.target_npos, task.target_epos, task.target_altitude
        
        delta_npos = (npos - target_npos) * 0.3048
        delta_epos = (epos - target_epos) * 0.3048
        delta_altitude = (altitude - target_altitude) * 0.3048
        
        reward_npos = torch.exp(-(delta_npos / self.error_scale) ** 2)
        reward_epos = torch.exp(-(delta_epos / self.error_scale) ** 2)
        reward_altitude = torch.exp(-(delta_altitude / self.error_scale) ** 2)
        
        reward_target = (reward_npos * reward_epos * reward_altitude) ** (1 / 3)
        
        cur_distances = torch.sqrt(
            (task.cur_npos - target_npos) ** 2 +
            (task.cur_epos - target_epos) ** 2 +
            (task.cur_altitude - target_altitude) ** 2
        )

        new_distances = torch.sqrt(
            (npos - target_npos) ** 2 +
            (epos - target_epos) ** 2 +
            (altitude - target_altitude) ** 2
        )
        
        distances_change = cur_distances - new_distances
        reward_distance = torch.where(
            distances_change > 0,
            torch.exp(0.1 * distances_change),
            torch.zeros_like(distances_change)
        )
        
        reward_distance = torch.clamp(reward_distance, max=self.max_reward)
        
        total_reward = reward_target + reward_distance
        
        return total_reward
        
        
    def get_reward_old(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        npos, epos, altitude = env.model.get_position()
        # delta_npos = (npos - task.target_npos) * 0.3048 / 1000              # unit: km
        # delta_epos = (epos - task.target_epos) * 0.3048 / 1000              # unit: km
        # delta_altitude = (altitude - task.target_altitude) * 0.3048 / 1000  # unit: km
        
        # Reach: 30.48m
        
        # reward_npos = -delta_npos ** 2
        # reward_epos = -delta_epos ** 2
        # reward_altitude = -delta_altitude ** 2
        # reward_target = reward_npos + reward_epos + reward_altitude
        # return 0.1 * reward_target
        
        # reward_npos = torch.exp(-(delta_npos ** 2))
        # reward_epos = torch.exp(-(delta_epos ** 2))
        # reward_altitude = torch.exp(-(delta_altitude ** 2))
        # reward_target = (reward_npos * reward_epos * reward_altitude) ** (1 / 3)
        
        delta_npos = (npos - task.target_npos) * 0.3048                 # unit: m (342, 762)~30
        delta_epos = (epos - task.target_epos) * 0.3048                 # unit: m 313~30
        delta_altitude = (altitude - task.target_altitude) * 0.3048     # unit: m 381~30
        
        npos_error_scale = 50  # 50m
        reward_npos = torch.exp(-(delta_npos / npos_error_scale) ** 2)
        
        epos_error_scale = 50  # 50m
        reward_epos = torch.exp(-(delta_epos / epos_error_scale) ** 2)
        
        altitude_error_scale = 50  # 50m
        reward_altitude = torch.exp(-(delta_altitude / altitude_error_scale) ** 2)
        
        reward_target = (reward_npos * reward_epos * reward_altitude) ** (1 / 3)
        
        cur_distances = self.cal_distance(
            task.cur_npos, task.cur_epos, task.cur_altitude,
            task.target_npos, task.target_epos, task.target_altitude
        )
        
        new_distances = self.cal_distance(
            npos, epos, altitude,
            task.target_npos, task.target_epos, task.target_altitude
        )
        
        distances_change = cur_distances - new_distances
        # 如果距离减少，给予指数奖励；否则奖励为 0
        scale = 0.1
        max_reward = 1
        reward_distance = torch.where(distances_change > 0, torch.exp(scale * distances_change), torch.zeros_like(distances_change))
        reward_distance = torch.clamp(reward_distance, max=max_reward)
        
        return reward_target + reward_distance
    
    def cal_distance(self, npos, epos, altitude, target_npos, target_epos, target_altitude):
        distances = torch.sqrt(
            (target_npos - npos) ** 2 +
            (target_epos - epos) ** 2 +
            (target_altitude - altitude) ** 2
        )
        
        return distances
        
