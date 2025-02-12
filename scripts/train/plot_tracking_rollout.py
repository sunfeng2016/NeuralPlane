import os
import pickle
import numpy as np
import matplotlib.pyplot as plt 


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



def draw_rollout(data, dirpath, _n = 10, _step = 10000):
    # for k in data.keys():
    #     # assert data[k].shape[0]==_step
    #     # assert data[k].shape[1]==_n
    #     data[k] = data[k, :10000]

    # columns = {'pitch':0,'head':1,'v':2}
    # columns = {'epos':0,'npos':1,'altitude':2}
    
    position_cols = {"npos": 0, "epos": 1, "altitude": 2}
    control_cols = {"pitch": 0, "heading": 1, "vt": 2}
    step = 10000

    for agent in range(_n):
        rollout_data = {}
        for position_k in position_cols.keys():
            rollout_data[position_k] = np.vstack([
                data['target_position'][:step, agent, position_cols[position_k]], 
                data['position'][:step, agent, position_cols[position_k]]
            ])
        for control_k in control_cols.keys():
            rollout_data[control_k] = np.vstack([
				data['target_control'][:step, agent, control_cols[control_k]] * 180 / np.pi, 
                data['real_control'][:step, agent, control_cols[control_k]] * 180 / np.pi
			])
        
        rollout_data['roll'] = np.vstack([data['roll'][:step, agent], data['roll'][:step,agent]]) 
        rollout_data['done'] = np.vstack([data['done'][:step, agent, 0], data['done'][:step,agent, 1]]) 

        # rollout_data['reward'] = np.vstack([data['reward'][:step, agent,0], data['reward'][:step,agent,0]]) 

        
        drawing_rollout_data(rollout_data, agent, dirpath)
        

if __name__ == "__main__":
    filename = "/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-13_17-56-40_Planning_tracking_F16_ppo_v10/250114_all_traj_data.pkl"
    dirpath = "/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/rollouts/"
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
        
    print(data)
    draw_rollout(data, dirpath, _n=10, _step=50000)