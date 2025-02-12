#!/usr/bin/env python
import sys
import os
import traceback
import datetime
import torch
import random
import logging
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from envs.planning_env import PlanningEnv
from algorithms.ppo.ppo_actor import PPOActor

from envs.utils.utils import wrap_PI, enu_to_geodetic

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcdefaults()
plt.rc('axes', unicode_minus=False)

CURRENT_WORK_PATH = os.getcwd()


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = False


def parse_args(args, parser):
    group = parser.add_argument_group("F16Sim Env parameters")
    group.add_argument("--env-name", type=str, default='PlanningEnv',
                       help="specify the name of environment")
    group.add_argument('--scenario-name', type=str, default='tracking',
                       help="Which scenario to run on")
    group.add_argument('--model-name', type=str, default='F16',
                       help="Which model to run on")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device(all_args.device)  # use cude mask to control using which GPU
        # torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        # torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figurepath = os.path.join(CURRENT_WORK_PATH, '..', f'./results/tracking/rollouts/{run_timestamp}')
    trackpath = os.path.join(CURRENT_WORK_PATH, '..',  f'./results/tracking/tracks/{run_timestamp}')
    datapath = os.path.join(CURRENT_WORK_PATH, '..',  f'./results/tracking/data/{run_timestamp}')
    
    os.makedirs(figurepath, exist_ok=True)
    os.makedirs(trackpath, exist_ok=True)
    
    _device = 'cuda:0'
    _n = 2200
    _step = 201

    # env init
    envs = PlanningEnv(num_envs=_n, config='tracking', model= 'F16', random_seed=42, device=_device, deterministic=True )

    config = {
        "all_args": all_args,
        "envs": envs,
        "device": device,
    }
    
    # _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-17_16-00-10_Planning_tracking_F16_ppo_v17/episode_300'
    
    # _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-21_11-14-00_Planning_tracking_F16_ppo_v20/episode_380'
    
    _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-22_13-41-37_Planning_tracking_F16_ppo_v22/episode_50'

    controller = PPOActor(config['all_args'], envs.observation_space, envs.action_space, device=_device)
    controller.eval()
    controller.load_state_dict(torch.load(_ego_run_dir + f"/actor_latest.pt"))
    ego_rnn_states = torch.zeros((_n, 1, 128), device=torch.device(_device))
    masks = torch.ones((_n, 1), device=_device)
    _obs = envs.reset()

    collect_data = {
        'obs':[],
        'target_position':[],
        'target_posture':[],
        'control_action':[],
        'track_action': [],
        'done':[],
        'reward':[]
    }
    
    for i in tqdm(range(_step)):                                         
        with torch.no_grad():
            actions, _, ego_rnn_states = controller(_obs, ego_rnn_states, masks, deterministic=True)

        _obs, reward, done, bad_done, exceed_time_limit, info = envs.step(actions)

        reset_env = (bad_done + exceed_time_limit).squeeze(axis=-1)
        ego_rnn_states[reset_env == True] = ego_rnn_states[reset_env == True]*0

        for k in info['collect_data'].keys():
            collect_data[k].append(info['collect_data'][k])
    
    for k in collect_data.keys():
        collect_data[k] = np.concatenate(collect_data[k],axis=0)

    import pickle
    os.makedirs(datapath, exist_ok=True)
    data_file = os.path.join(datapath, f"{run_timestamp}_all_traj.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(collect_data, f)
    
            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
