#!/usr/bin/env python
import sys
import os
import datetime
import torch
import random
import logging
import numpy as np
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
    
    figurepath = os.path.join(CURRENT_WORK_PATH, '..', f'./results/control/rollouts/{run_timestamp}')
    trackpath = os.path.join(CURRENT_WORK_PATH, '..',  f'./results/control/tracks/{run_timestamp}')
    datapath = os.path.join(CURRENT_WORK_PATH, '..',  f'./results/control/data/{run_timestamp}')
    
    _device = 'cuda:0'
    _n = 10
    _step = 1000
    
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
        'target':[],
		'obs':[],
		'action':[], 
		'terminated':[],
		'reward':[],
		'roll': [],
		'npos': [],
		'epos': [],
		'altitude': [],
        "T": [],
        "el": [],
        "ail": [],
        "rud": [], 
        "ax": [],
        "ay": [],
        "az": [],
        "acceleration": [],  
	}
    
    for i in range(_step):
        target_pitch = envs.task.target_pitch * 180 / np.pi
        target_heading = envs.task.target_heading * 180 / np.pi
        target_vt = envs.task.target_vt * 0.3048
        
        collect_data['target'].append(torch.cat([target_pitch.unsqueeze(-1), 
                                                 target_heading.unsqueeze(-1), 
                                                 target_vt.unsqueeze(-1)], dim=-1).detach().cpu().numpy())
        
        with torch.no_grad():
            actions, _, ego_rnn_states = controller(_obs, ego_rnn_states, masks, deterministic=True)
        _obs, reward, done, bad_done, exceed_time_limit, info = envs.step(actions)
            
        roll, pitch, heading = envs.model.get_posture()
        roll, pitch, heading = wrap_PI(roll) * 180 / np.pi, wrap_PI(pitch) * 180 / np.pi, wrap_PI(heading) * 180 / np.pi
        
        vt = envs.model.get_vt() * 0.3048
        collect_data['obs'].append(torch.cat([pitch.unsqueeze(-1), 
                                              heading.unsqueeze(-1), 
                                              vt.unsqueeze(-1)],dim=-1).detach().cpu().numpy())
        
        collect_data["roll"].append(roll.detach().cpu().numpy())
        
        npos, epos, altitude = envs.model.get_position()
        npos, epos, altitude = npos * 0.3048, epos * 0.3048, altitude * 0.3048
        
        collect_data["npos"].append(npos.detach().cpu().numpy())
        collect_data["epos"].append(epos.detach().cpu().numpy())
        collect_data["altitude"].append(altitude.detach().cpu().numpy())
        
        reset_env = (bad_done + exceed_time_limit).squeeze(axis=-1)
        ego_rnn_states[reset_env == True] = ego_rnn_states[reset_env == True]*0
        
        collect_data['action'].append(actions.detach().cpu().numpy())
        collect_data['terminated'].append(torch.cat([done.unsqueeze(-1), 
                                                     bad_done.unsqueeze(-1),
                                                     exceed_time_limit.unsqueeze(-1)],dim=-1).detach().cpu().numpy())
        collect_data['reward'].append(reward.unsqueeze(-1).detach().cpu().numpy())
        
        T = envs.model.get_thrust()
        el, ail, rud, lef = envs.model.get_control_surface()
        ax, ay, az = envs.model.get_acceleration()
        acceleration = ax ** 2 + ay ** 2 + az ** 2
        acceleration = torch.sqrt(acceleration)
        
        collect_data['T'].append(_t2n(T))
        collect_data['el'].append(_t2n(el))
        collect_data['ail'].append(_t2n(ail))
        collect_data['rud'].append(_t2n(rud))
        
        collect_data['ax'].append(_t2n(ax))
        collect_data['ay'].append(_t2n(ay))
        collect_data['az'].append(_t2n(az))
        
        collect_data['acceleration'].append(_t2n(acceleration))
        
    for k in collect_data.keys():
        collect_data[k] = np.stack(collect_data[k])

    draw_rollout(collect_data, figurepath, _n = _n, _step = _step)
    render_rollout(collect_data, trackpath, _n = _n, _step = _step)
    
def save_data(data, episode, datapath):
    os.makedirs(datapath, exist_ok=True)
    file_path = os.path.join(datapath, f'episode_{episode}.pkl')
    
    import pickle
    with open(file_path, "wb") as file:
        pickle.dump(data, file)
    
    logging.info(f'episode_{episode}.pkl has been saved to \n {file_path}')
    
    
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

def draw_rollout(data, dirpath, _n = 10, _step = 3000):
    for k in data.keys():
        assert data[k].shape[0]==_step
        assert data[k].shape[1]==_n

    columns = {'pitch':0,'head':1,'v':2}

    for agent in range(_n):
        rollout_data = {}
        for col_k in columns.keys():
            rollout_data[col_k] = np.vstack([data['target'][:,agent,columns[col_k]], data['obs'][:,agent,columns[col_k]]]) 
        rollout_data['reward'] = np.vstack([data['reward'][:,agent,0], data['reward'][:,agent,0]]) 
        # rollout_data['roll'] = np.vstack([data['roll'][:,agent], data['roll'][:,agent]])
        # rollout_data['epos'] = np.vstack([data['epos'][:,agent], data['epos'][:,agent]])
        # rollout_data['npos'] = np.vstack([data['npos'][:,agent], data['npos'][:,agent]])
        # rollout_data['altitude'] = np.vstack([data['altitude'][:,agent], data['altitude'][:,agent]])
        rollout_data['T'] = np.vstack([data['T'][:,agent], data['action'][:,agent,0] * 0.225 * 76300 / 0.3048])
        rollout_data['el'] = np.vstack([data['el'][:,agent], data['action'][:,agent,1] * 45])
        rollout_data['ail'] = np.vstack([data['ail'][:,agent], data['action'][:,agent,2] * 45])
        rollout_data['rud'] = np.vstack([data['rud'][:,agent], data['action'][:,agent,3] * 45])
        
        rollout_data['ax'] = np.vstack([data['ax'][:,agent], data['ax'][:,agent]])
        rollout_data['ay'] = np.vstack([data['ay'][:,agent], data['ay'][:,agent]])
        rollout_data['az'] = np.vstack([data['az'][:,agent], data['az'][:,agent]])
        rollout_data['acceleration'] = np.vstack([data['acceleration'][:,agent], data['acceleration'][:,agent]])
        
        drawing_rollout_data(rollout_data, agent, dirpath)
        
def render_rollout(data, dirpath, _n=10, _step=3000):
    columns = {'pitch':0,'head':1,'v':2}
    
    for agent in range(_n):
        render_data = {}
        for col_k in columns.keys():
            render_data[col_k] = data['obs'][:, agent, columns[col_k]]
        render_data['roll'] = data['roll'][:, agent]
        render_data['epos'] = data['epos'][:, agent]
        render_data['npos'] = data['npos'][:, agent]
        render_data['altitude'] = data['altitude'][:, agent]
        render_rollout_data(render_data, agent, dirpath, _step)
            
def render_rollout_data(render_data, traj_id, dirpath, total_step):
    os.makedirs(dirpath, exist_ok=True)
    filename = os.path.join(dirpath,f'{traj_id}.txt.acmi')
    
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.0\n")
        f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")
        
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
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
