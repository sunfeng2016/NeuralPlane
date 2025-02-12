#!/usr/bin/env python
import sys
import os
import datetime
import torch
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from envs.control_env import ControlEnv
from algorithms.ppo.ppo_actor import PPOActor

from envs.utils.utils import wrap_PI, enu_to_geodetic, _t2n

import matplotlib.pyplot as plt
plt.rcdefaults()
plt.rc('axes', unicode_minus=False)

CURRENT_WORK_PATH = os.getcwd()

def parse_args(args, parser):
    group = parser.add_argument_group("F16Sim Env parameters")
    group.add_argument("--env-name", type=str, default='Control',
                       help="specify the name of environment")
    group.add_argument('--scenario-name', type=str, default='control',
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
    
    datapath = os.path.join(CURRENT_WORK_PATH, '..',  f'./results/control/data/{run_timestamp}')
    
    _device = 'cuda:0'
    _n = 3000
    _step = 2501
    
    # env init
    envs = ControlEnv(num_envs=_n, config='control', random_seed=42, device=_device, deterministic=True)

    config = {
        "all_args": all_args,
        "envs": envs,
        "device": device,
    }
    _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-10_11-49-15_Control_control_F16_ppo_v6/episode_249'
    controller = PPOActor(config['all_args'], envs.observation_space, envs.action_space, device=_device)
    controller.eval()
    controller.load_state_dict(torch.load(_ego_run_dir + f"/actor_latest.pt"))
    ego_rnn_states = torch.zeros((_n, 1, 128), device=torch.device(_device))
    masks = torch.ones((_n, 1), device=_device)
    _obs = envs.reset()

    collect_data = {
        'obs': [],
        'next_obs': [],
        'action': [],
		'reward': [],
		'done': [],
		'bad_done': [],
		'exceed_time_limit': []              
	}
    
    for i in tqdm(range(_step)):
        collect_data['obs'].append(_t2n(_obs))
        with torch.no_grad():
            actions, _, ego_rnn_states = controller(_obs, ego_rnn_states, masks, deterministic=True)
        _obs, reward, done, bad_done, exceed_time_limit, info = envs.step(actions)
        
        reset_env = (bad_done + exceed_time_limit).squeeze(axis=-1)
        ego_rnn_states[reset_env == True] = ego_rnn_states[reset_env == True]*0
        
        collect_data['next_obs'].append(_t2n(_obs))
        collect_data['action'].append(_t2n(actions))
        collect_data['reward'].append(_t2n(reward))
        collect_data['done'].append(_t2n(done))
        collect_data['bad_done'].append(_t2n(bad_done))
        collect_data['exceed_time_limit'].append(_t2n(exceed_time_limit))

    for k in collect_data.keys():
        collect_data[k] = np.stack(collect_data[k])
        
    save_data(collect_data, datapath, _n, _step)
        
def save_data(data, datapath, n, step):
    os.makedirs(datapath, exist_ok=True)
    for k in data.keys():
        assert data[k].shape[0] == step
        assert data[k].shape[1] == n
    
    for agent in range(n):
        rollout_data = {}
        length = np.where(data['bad_done'][:, agent] | data['exceed_time_limit'][:, agent])[0][0] + 1
        print(f"The length of trajectory-{agent} is {length}")
        rollout_data['obs'] = data['obs'][:length, agent, :]
        rollout_data['next_obs'] = data['next_obs'][:length, agent, :]
        rollout_data['action'] = data['action'][:length, agent, :]
        rollout_data['reward'] = data['reward'][:length, agent]
        rollout_data['done'] = data['done'][:length, agent]
        rollout_data['bad_done'] = data['bad_done'][:length, agent]
        rollout_data['exceed_time_limit'] = data['exceed_time_limit'][:length, agent]
        
        data_file = os.path.join(datapath, f'episode_{agent}.pkl')
        with open(data_file, 'wb') as f:
            pickle.dump(rollout_data, f)
        
        logging.info(f'episode_{agent}.pkl has been saved to \n {data_file}')
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
