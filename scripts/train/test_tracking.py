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
    _n = 1
    _step = 1000

    # env init
    envs = PlanningEnv(num_envs=_n, config='tracking', model= 'F16', random_seed=42, device=_device, deterministic=True )

    config = {
        "all_args": all_args,
        "envs": envs,
        "device": device,
    }
    
    # _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-17_16-00-10_Planning_tracking_F16_ppo_v17/episode_300'
    
    # _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-21_11-14-00_Planning_tracking_F16_ppo_v20/episode_380'
    
    # _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-22_13-41-37_Planning_tracking_F16_ppo_v22/episode_200'
    
    # _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-21_11-14-00_Planning_tracking_F16_ppo_v20/episode_50'
    
    _ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-22_13-41-37_Planning_tracking_F16_ppo_v22/episode_50'

    controller = PPOActor(config['all_args'], envs.observation_space, envs.action_space, device=_device)
    controller.eval()
    controller.load_state_dict(torch.load(_ego_run_dir + f"/actor_latest.pt"))
    ego_rnn_states = torch.zeros((_n, 1, 128), device=torch.device(_device))
    masks = torch.ones((_n, 1), device=_device)
    _obs = envs.reset()

    collect_data = {
    	"position": [],
		"target_position": [],
		"real_control": [],
		"target_control": [],
        "low_action": [],
		"roll": [],
		"done": [],
		"reward": [],
        "Overload": [],
        "ExtremeState": [],
        "ReachTarget": [],
        "UnreachTarget": [], 
        "alpha": [],
        "beta": [],
        "ax": [],
        "ay": [],            
        "az": [],  
        "T": [],
        "el": [],
        "ail": [],
        "rud": []                      
    }
    
    for i in tqdm(range(_step)):                                         
        with torch.no_grad():
            actions, _, ego_rnn_states = controller(_obs, ego_rnn_states, masks, deterministic=True)

        _obs, reward, done, bad_done, exceed_time_limit, info = envs.step(actions)

        reset_env = (bad_done + exceed_time_limit).squeeze(axis=-1)
        ego_rnn_states[reset_env == True] = ego_rnn_states[reset_env == True]*0

        for k in info['mini_data'].keys():
            collect_data[k].append(info['mini_data'][k])
    
    for k in collect_data.keys():
        collect_data[k] = np.concatenate(collect_data[k],axis=0)

    import pickle
    os.makedirs(datapath, exist_ok=True)
    data_file = os.path.join(datapath, "all_traj.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(collect_data, f)

    total_step = collect_data['done'].shape[0]
    # total_step = 2000
    # draw_rollout(collect_data, figurepath, n=_n, step=total_step)
    render_rollout(collect_data, trackpath, n=_n, step=total_step)
    
def draw_rollout(data, dirpath, n=10, step=10000):
    position_cols = {"npos": 0, "epos": 1, "altitude": 2}
    control_cols = {"pitch": 0, "heading": 1, "vt": 2}
    
    for agent in range(n):
        rollout_data = {}
        for position_k in position_cols.keys():
            rollout_data[position_k] = np.vstack([
                data['target_position'][:step, agent, position_cols[position_k]], 
                data['position'][:step, agent, position_cols[position_k]]
            ])
        for control_k in control_cols.keys():
            rollout_data[control_k] = np.vstack([
				data['target_control'][:step, agent, control_cols[control_k]], 
                data['real_control'][:step, agent, control_cols[control_k]]
			])
        
        # rollout_data['roll'] = np.vstack([data['roll'][:step, agent], data['roll'][:step,agent]]) 
        # rollout_data['reward'] = np.vstack([data['reward'][:step, agent, 0], data['reward'][:step,agent, 0]]) 
        # rollout_data['alpha'] = np.vstack([data['alpha'][:step, agent, 0], data['alpha'][:step,agent, 0]]) 
        # rollout_data['beta'] = np.vstack([data['beta'][:step, agent, 0], data['beta'][:step,agent, 0]]) 
        rollout_data['ax'] = np.vstack([data['ax'][:step, agent, 0], data['ax'][:step,agent, 0]]) 
        rollout_data['ay'] = np.vstack([data['ay'][:step, agent, 0], data['ay'][:step,agent, 0]]) 
        rollout_data['az'] = np.vstack([data['az'][:step, agent, 0], data['az'][:step,agent, 0]]) 
        
        rollout_data['T'] = np.vstack([data['T'][:step, agent, 0], data['T'][:step,agent, 0]]) 
        rollout_data['el'] = np.vstack([data['el'][:step, agent, 0], data['T'][:step,agent, 0]]) 
        rollout_data['ail'] = np.vstack([data['ail'][:step, agent, 0], data['T'][:step,agent, 0]]) 
        rollout_data['rud'] = np.vstack([data['rud'][:step, agent, 0], data['T'][:step,agent, 0]]) 
        
        rollout_data['done'] = np.vstack([data['done'][:step, agent, 1] * -1, data['done'][:step,agent, 0]]) 
        
        # rollout_data['Overload'] = np.vstack([data['Overload'][:step, agent], data['Overload'][:step,agent]]) 
        # rollout_data['ExtremeState'] = np.vstack([data['ExtremeState'][:step, agent], data['ExtremeState'][:step,agent]]) 
        # rollout_data['ReachTarget'] = np.vstack([data['ReachTarget'][:step, agent], data['ReachTarget'][:step,agent]]) 
        # rollout_data['UnreachTarget'] = np.vstack([data['UnreachTarget'][:step, agent], data['UnreachTarget'][:step,agent]]) 
        
        drawing_rollout_data(rollout_data, agent, dirpath)
        
def drawing_rollout_data(drawdata, traj_id, dirpath):
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath,f'{traj_id}.png')

    fig, axs = plt.subplots(len(drawdata), 1, figsize=(15, len(drawdata)*5.5))  # figsize 设置宽15, 高5 * 15 = 75

    for idx, (key, data) in enumerate(drawdata.items()):
        axs[idx].set_ylabel(key)
        axs[idx].set_xlabel('step')
        axs[idx].plot(data[0], label='target')
        axs[idx].plot(data[1], label='obs')
        axs[idx].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)

    plt.close()
    
def render_rollout(data, dirpath, n=10, step=10000):
    position_cols = {"npos": 0, "epos": 1, "altitude": 2}
    control_cols = {"pitch": 0, "head": 1, "vt": 2}
    
    for agent in range(n):
        rollout_data = {}
        for position_k in position_cols.keys():
            rollout_data[position_k] = data['position'][:step, agent, position_cols[position_k]]
            rollout_data[f'target_{position_k}'] = data['target_position'][:step, agent, position_cols[position_k]]
        for control_k in control_cols.keys():
            rollout_data[control_k] = data['real_control'][:step, agent, control_cols[control_k]]
        
        rollout_data['roll'] = data['roll'][:step, agent]
        rollout_data['done'] = data['done'][:step, agent, 0]
        rollout_data['bad_done'] = data['done'][:step, agent, 1]
        rollout_data['time_out'] = data['done'][:step, agent, 2]
        
        render_rollout_data(rollout_data, agent, dirpath, step)
    
def render_rollout_data(render_data, traj_id, dirpath, total_step):
    os.makedirs(dirpath, exist_ok=True)
    filename = os.path.join(dirpath,f'{traj_id}.txt.acmi')
    
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.0\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")
        
        num_target = 0
        new_target = True
        
        for step in range(total_step):
            timestamp = step * 0.02
            f.write(f"#{timestamp:.2f}\n")
            npos = render_data['npos'][step]
            epos = render_data['epos'][step]
            alt = render_data['altitude'][step]
            roll = render_data['roll'][step]
            pitch = render_data['pitch'][step]
            yaw = render_data['head'][step]
            lat, lon, alt = enu_to_geodetic(epos, npos, alt, 0, 0, 0)
            log_msg = f"{100 + traj_id},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Name=F16,"
            log_msg += f"Color=Red"
            if log_msg is not None:
                f.write(log_msg + "\n")
            
            if new_target:
                new_target = False
                target_npos = render_data['target_npos'][step]
                target_epos = render_data['target_epos'][step]
                target_alt = render_data['target_altitude'][step]
                target_lat, target_lon, target_alt = enu_to_geodetic(target_epos, target_npos, target_alt, 0, 0, 0)
                log_msg_temp = f"{10000}, T={target_lon}|{target_lat}|{target_alt},Name=Target,Color=Blue"
                if log_msg_temp is not None:
                    f.write(log_msg_temp + "\n")
            
            done = render_data['done'][step]
            bad_done = render_data['bad_done'][step]
            time_out = render_data['time_out'][step]
                    
            if done | bad_done | time_out:
                num_target += 1
                new_target = True
    
import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # 如果您希望输出立即可见
    def flush(self) :
        for f in self.files:
            f.flush()
            
output_file = "./output_log.txt"
            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        main(sys.argv[1:])
        sys.stdout = original_stdout

