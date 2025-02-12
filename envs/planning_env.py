import sys
import os
import gym
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from env_base import BaseEnv
from models.F16_model import F16Model
from models.UAV_model import UAVModel
from tasks.tracking_task import TrackingTask
from utils.utils import wrap_PI
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.ppo.ppo_actor import PPOActor
from utils.utils import wrap_PI

CURRENT_WORK_PATH = os.getcwd()
CURRENT_WORK_PATH = "/home/ubuntu/sunfeng/MARL/NeuralPlane/envs/"
# ego_run_dir = CURRENT_WORK_PATH + "/../scripts/runs/2024-12-06_15-59-14_Control_control_F16_ppo_v2/episode_249"
ego_run_dir = CURRENT_WORK_PATH + "/../scripts/runs/2024-12-31_11-52-57_Control_control_F16_ppo_v3/episode_249"
ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-08_18-04-43_Control_control_F16_ppo_v4/episode_200'
ego_run_dir = '/home/ubuntu/sunfeng/MARL/NeuralPlane/scripts/runs/2025-01-10_11-49-15_Control_control_F16_ppo_v6/episode_249'

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

class PlanningEnv(BaseEnv):
    """
    PlanningEnv is a fly-planning env for single agent to do tracking task.
    """
    def __init__(self, num_envs=1, config='tracking', model='F16', random_seed=None, device="cuda:0", deterministic=False):
        super().__init__(num_envs, config, model, random_seed, device)
        self.deterministic = deterministic
        self.low_level_action_space = gym.spaces.Box(low=-np.inf,
                                                     high=np.inf,
                                                     shape=(4, ))
        args = Args()
        self.controller = PPOActor(args, self.observation_space, self.low_level_action_space, device=self.device)
        self.controller.eval()
        self.controller.load_state_dict(torch.load(ego_run_dir + f"/actor_latest.pt", map_location=torch.device('cuda:0')))
        self.ego_rnn_states = torch.zeros((self.n, 1, 128), device=torch.device(device))
    
    def info(self):
        return {"done": 
            {
                "Overload": 0,
                "LowAltitude": 0,
                "HighSpeed": 0,
                "LowSpeed": 0,
                "ExtremeState": 0,
                "UnreachPosture": 0,
                "ReachPosture": 0,
                "Timeout": 0,
            }, 
            "step": self.total_step,
            "termination": {
                "Overload": np.zeros(self.n, dtype=bool),
                "ExtremeState": np.zeros(self.n, dtype=bool),
                "ReachTarget": np.zeros(self.n, dtype=bool),
                "UnreachTarget": np.zeros(self.n, dtype=bool),
            }
        }

    def load(self, random_seed, config, model):
        if random_seed is not None:
            self.seed(random_seed)
        if model == 'F16':
            self.model = F16Model(self.config, self.n, self.device, random_seed)
        elif model == 'UAV':
            self.model = UAVModel(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
        if config == 'tracking':
            self.task = TrackingTask(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
    
    def low_level_obs(self, target_pitch, target_heading, target_vt):
        """
        Convert actions into the format of observation_space of low level controller.

        observation(dim 22):
            0. ego_delta_pitch      (unit: rad)
            1. ego_delta_heading       (unit rad)
            2. ego_delta_vt            (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
            16. ego_T                  (unit: %)
            17. ego_el                 (unit: %)
            18. ego_ail                (unit: %)
            19. ego_rud                (unit: %)
            20. ego_lef                (unit: %)
            21. EAS2TAS
        """
        npos, epos, altitude = self.model.get_position()
        roll, pitch, heading = self.model.get_posture()
        vt = self.model.get_vt()
        EAS = self.model.get_EAS()
        alpha = self.model.get_AOA()
        beta = self.model.get_AOS()
        P, Q, R = self.model.get_angular_velocity()
        T = self.model.get_thrust()
        el, ail, rud, lef = self.model.get_control_surface()
        eas2tas = self.model.get_EAS2TAS()

        norm_delta_pitch = wrap_PI((pitch - target_pitch).reshape(-1, 1))
        norm_delta_heading = wrap_PI((heading - target_heading).reshape(-1, 1))
        norm_delta_vt = (vt - target_vt).reshape(-1, 1) * 0.3048 / 340
        norm_altitude = altitude.reshape(-1, 1) * 0.3048 / 5000
        roll_sin = torch.sin(roll.reshape(-1, 1))
        roll_cos = torch.cos(roll.reshape(-1, 1))
        pitch_sin = torch.sin(pitch.reshape(-1, 1))
        pitch_cos = torch.cos(pitch.reshape(-1, 1))
        # norm_vt = vt.reshape(-1, 1) * 0.3048 / 340
        norm_EAS = EAS.reshape(-1, 1) * 0.3048 / 340
        alpha_sin = torch.sin(alpha.reshape(-1, 1))
        alpha_cos = torch.cos(alpha.reshape(-1, 1))
        beta_sin = torch.sin(beta.reshape(-1, 1))
        beta_cos = torch.cos(beta.reshape(-1, 1))
        norm_P = P.reshape(-1, 1)
        norm_Q = Q.reshape(-1, 1)
        norm_R = R.reshape(-1, 1)
        norm_T = T.reshape(-1, 1) / 0.225 / 76300 * 0.3048
        norm_el = el.reshape(-1, 1) / 45
        norm_ail = ail.reshape(-1, 1) / 45
        norm_rud = rud.reshape(-1, 1) / 45
        norm_lef = lef.reshape(-1, 1) / 45
        obs = torch.hstack((norm_delta_pitch, norm_delta_heading))
        obs = torch.hstack((obs, norm_delta_vt))
        obs = torch.hstack((obs, norm_altitude))
        obs = torch.hstack((obs, roll_sin))
        obs = torch.hstack((obs, roll_cos))
        obs = torch.hstack((obs, pitch_sin))
        obs = torch.hstack((obs, pitch_cos))
        obs = torch.hstack((obs, norm_EAS))
        obs = torch.hstack((obs, alpha_sin))
        obs = torch.hstack((obs, alpha_cos))
        obs = torch.hstack((obs, beta_sin))
        obs = torch.hstack((obs, beta_cos))
        obs = torch.hstack((obs, norm_P))
        obs = torch.hstack((obs, norm_Q))
        obs = torch.hstack((obs, norm_R))
        obs = torch.hstack((obs, norm_T))
        obs = torch.hstack((obs, norm_el))
        obs = torch.hstack((obs, norm_ail))
        obs = torch.hstack((obs, norm_rud))
        obs = torch.hstack((obs, norm_lef))
        obs = torch.hstack((obs, eas2tas.reshape(-1, 1)))
        return obs
    
    def step_0108(self, action, render=False, count=0):
        self.reset()
        action = torch.clamp(action, -1, 1)
        # set target
        roll, pitch, yaw = self.model.get_posture()
        vt = self.model.get_vt()
        target_pitch = pitch + action[:, 0] * 0.3
        target_heading = yaw + action[:, 1] * 0.3
        target_vt = vt + action[:, 2] * 30
        
        if self.deterministic:
            print("*" * 100)
            npos, epos, altitude = self.model.get_position()
            print(f"position: ({npos.item()}, {epos.item()}, {altitude.item()}), target: ({self.task.target_npos.item()}, {self.task.target_epos.item()}, {self.task.target_altitude.item()})")
            
        for i in range(50):
            # low-level control
            ego_obs = self.low_level_obs(target_pitch, target_heading, target_vt)
            masks = torch.ones((self.n, 1), device=self.device)
            with torch.no_grad():
                ego_actions, _, self.ego_rnn_states = self.controller(ego_obs, self.ego_rnn_states, masks, deterministic=True)
            
            # step
            self.model.update(ego_actions)
            done = self.is_done.bool()
            bad_done = self.bad_done.bool()
            exceed_time_limit = self.exceed_time_limit.bool()
            reset = (done | bad_done) | exceed_time_limit
            self.model.s[reset] = self.model.recent_s[reset]
            self.model.u[reset] = self.model.recent_u[reset]
            
            # self.step_count += 1
            obs = self.obs()
            info = self.info()
            done, bad_done, exceed_time_limit, info = self.done(info)
            reward = self.reward()

            if render:
                self.render(count=self.total_step)
            # count += 1
            self.total_step += 1
            
            if self.deterministic:
                print(f"step: {count}, meta-step: {i}, ego_action: {ego_actions}")
                if torch.all(reset):
                    break
        
        self.step_count += 1
        
        return obs, reward, done, bad_done, exceed_time_limit, info
    
    def get_position(self):
        npos, epos, altitude = self.model.get_position()
        postion = torch.stack([npos * 0.3048, epos * 0.3048, altitude * 0.3048], dim=-1).detach().cpu().numpy()
        target_position = torch.stack([self.task.target_npos * 0.3048, self.task.target_epos * 0.3048, self.task.target_altitude * 0.3048], dim=-1).detach().cpu().numpy() 
        
        return postion, target_position
    
    def get_control(self, target_pitch, target_heading, target_vt):
        target_control = torch.stack([target_pitch * 180 / np.pi, target_heading * 180 / np.pi, target_vt * 0.3048], dim=-1).detach().cpu().numpy()
        roll, pitch, heading = self.model.get_posture()
        roll, pitch, heading = wrap_PI(roll) * 180 / np.pi, wrap_PI(pitch) * 180 / np.pi, wrap_PI(heading) * 180 / np.pi
        vt = self.model.get_vt() * 0.3048
        real_control = torch.stack([pitch , heading, vt], dim=-1).detach().cpu().numpy()
        
        return real_control, target_control, roll.detach().cpu().numpy()
    
    def mini_get_data(self, env):
        tp_npos, tp_epos, tp_altitude = env.task.target_npos* 0.3048, env.task.target_epos* 0.3048, env.task.target_altitude* 0.3048

        npos, epos, altitude = env.model.get_position()
        npos, epos, altitude = npos* 0.3048, epos* 0.3048, altitude* 0.3048
        
        roll, pitch, heading = env.model.get_posture()
        roll, pitch, heading = wrap_PI(roll), wrap_PI(pitch), wrap_PI(heading)
        
        alpha = env.model.get_AOA()
        beta = env.model.get_AOS()
        alpha, beta = wrap_PI(alpha), wrap_PI(beta)        
        
        P, Q, R = env.model.get_angular_velocity()

        T = env.model.get_thrust()/ 0.225 / 76300 * 0.3048
        el, ail, rud, lef = env.model.get_control_surface()

        vt = env.model.get_vt()* 0.3048
        EAS = env.model.get_EAS() * 0.3048
        eas2tas = env.model.get_EAS2TAS()
        
        vel_u, vel_v, vel_w = env.model.get_velocity()
        vel_u, vel_v, vel_w = vel_u * 0.3048, vel_v * 0.3048, vel_w * 0.3048

        _tp_position = torch.stack([tp_epos, tp_npos,  tp_altitude],dim=-1).detach().cpu().numpy()
        _obs = torch.stack([epos, npos,  altitude, roll, pitch, heading, alpha, beta, P, Q, R, T, el, ail, rud, lef, vt, vel_u, vel_v, vel_w, EAS, eas2tas],dim=-1).detach().cpu().numpy()

        return _tp_position, _obs
    
    def get_target(self):
        cur_npos = self.task.cur_npos
        cur_epos = self.task.cur_epos
        cur_altitude = self.task.cur_altitude
        
        target_npos = self.task.target_npos
        target_epos = self.task.target_epos
        target_altitude = self.task.target_altitude
        
        roll, pitch, yaw = self.model.get_posture()
        
        # 计算目标方向向量
        direction_vector = torch.stack([target_npos-cur_npos, target_epos-cur_epos, target_altitude-cur_altitude], dim=1) # (N, 3)
        direction_norm = torch.norm(direction_vector, dim=1, keepdim=True) # (N, 1)
        direction_unit = direction_vector / direction_norm # (N, 3)
        
        # 计算目标偏航角
        yaw_target = torch.atan2(direction_unit[:, 1], direction_unit[:, 0]) 
        
        # 计算目标俯仰角
        pitch_target = torch.asin(direction_unit[:, 2])
        
        # 计算delta
        delta_yaw = wrap_PI(yaw_target - yaw)
        delta_pitch = wrap_PI(pitch_target - pitch)
        
        # 限制调整量在 [-0.3, 0.3] 范围内
        delta_yaw = torch.clamp(delta_yaw, -0.3, 0.3)
        delta_pitch = torch.clamp(delta_pitch, -0.3, 0.3)
        
        return delta_pitch, delta_yaw
        
    def step(self, action, render=False, count=0):
        self.reset()
        action = torch.clamp(action, -1, 1)
        # get position
        self.task.cur_npos, self.task.cur_epos, self.task.cur_altitude = self.model.get_position()
        # set target
        roll, pitch, yaw = self.model.get_posture()
        target_pitch = wrap_PI(pitch + action[:, 0] * 0.3)
        target_heading = wrap_PI(yaw + action[:, 1] * 0.3)
        vt = self.model.get_vt()
        target_vt = vt + action[:, 2] * 30
        
        if self.deterministic:
            mini_data = {
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
            
            collect_data = {
                'obs':[],                   # 态势信息 [epos, npos,  altitude, roll, pitch, heading, alpha, beta, P, Q, R, T, el, ail, rud, lef, vt, vel_u, vel_v, vel_w, EAS, eas2tas]
                'target_position':[],       # 目标位置 (N, E, A)
                'target_posture':[],        # 目标姿态 (vt, pitch, heading)
                'control_action':[],        # 底层动作 (T, el, ail, rud, lef)
                'track_action': [],         # 上层动作 (delta_pitch, delta_heading, delta_vt)
                'done':[],
                'reward':[]
            }
        
        # tracking step
        for i in range(50):
            
            if self.deterministic:
                position, target_position = self.get_position()
                mini_data["position"].append(position)
                mini_data["target_position"].append(target_position)
                
                _tp_position, _obs = self.mini_get_data(self)
                collect_data['obs'].append(_obs)
                collect_data['target_position'].append(_tp_position)
                collect_data['target_posture'].append(torch.stack([target_vt, target_pitch,  target_heading],dim=-1).detach().cpu().numpy())
            
            # low-level control
            ego_obs = self.low_level_obs(target_pitch, target_heading, target_vt)
            masks = torch.ones((self.n, 1), device=self.device)
            with torch.no_grad():
                ego_actions, _, self.ego_rnn_states = self.controller(ego_obs, self.ego_rnn_states, masks, deterministic=True)
            
            # control step
            self.model.update(ego_actions)
            
            if self.deterministic:
                real_control, target_control, roll = self.get_control(target_pitch, target_heading, target_vt)
                mini_data['real_control'].append(real_control)
                mini_data['target_control'].append(target_control)
                mini_data['roll'].append(roll)
            
            # wait
            done = self.is_done.bool()
            bad_done = self.bad_done.bool()
            exceed_time_limit = self.exceed_time_limit.bool()
            reset = (done | bad_done) | exceed_time_limit
            self.model.s[reset] = self.model.recent_s[reset]
            self.model.u[reset] = self.model.recent_u[reset]
            
            # obs = self.obs()
            info = self.info()
            done, bad_done, exceed_time_limit, info = self.done(info)
            # reward = self.reward() * 0.1
            reward = torch.zeros_like(done)
            
            alpha = self.model.get_AOA() * 180 / torch.pi
            beta = self.model.get_AOS() * 180 / torch.pi
            
            ax, ay, az = self.model.get_acceleration()
            T = self.model.get_thrust()
            el, ail, rud, lef = self.model.get_control_surface()
            
            if self.deterministic:
                mini_data['done'].append(torch.stack([done, bad_done, exceed_time_limit],dim=-1).detach().cpu().numpy())
                mini_data['reward'].append(torch.stack([reward],dim=-1).detach().cpu().numpy())
                mini_data['alpha'].append(torch.stack([alpha],dim=-1).detach().cpu().numpy())
                mini_data['beta'].append(torch.stack([beta],dim=-1).detach().cpu().numpy())
                mini_data['ax'].append(torch.stack([ax],dim=-1).detach().cpu().numpy())
                mini_data['ay'].append(torch.stack([ay],dim=-1).detach().cpu().numpy())
                mini_data['az'].append(torch.stack([az],dim=-1).detach().cpu().numpy())
                mini_data['low_action'].append(ego_actions.detach().cpu().numpy())
                mini_data['T'].append(torch.stack([T],dim=-1).detach().cpu().numpy())
                mini_data['el'].append(torch.stack([el],dim=-1).detach().cpu().numpy())
                mini_data['ail'].append(torch.stack([ail],dim=-1).detach().cpu().numpy())
                mini_data['rud'].append(torch.stack([rud],dim=-1).detach().cpu().numpy())
                
                mini_data['Overload'].append(info['termination']['Overload'])
                mini_data['ExtremeState'].append(info['termination']['ExtremeState'])
                mini_data['ReachTarget'].append(info['termination']['ReachTarget'])
                mini_data['UnreachTarget'].append(info['termination']['UnreachTarget'])
                
                collect_data['track_action'].append(action.detach().cpu().numpy())
                collect_data['control_action'].append(ego_actions.detach().cpu().numpy())
                collect_data['done'].append(torch.stack([done, bad_done, exceed_time_limit],dim=-1).detach().cpu().numpy())
                collect_data['reward'].append(torch.stack([reward],dim=-1).detach().cpu().numpy())

            if render:
                self.render(count=self.total_step)
            
            self.total_step += 1
            self.step_count += 1
            self.episode_step += 1

        obs = self.obs()
        # reward = self.reward() * 0.1
        reward = self.reward() * 0.1
        # print('-'*20)
        # print(reward)
        # print(torch.mean(reward))
        
        if self.deterministic:
            mini_data['reward'][-1] = torch.stack([reward],dim=-1).detach().cpu().numpy()
            collect_data['reward'][-1] = torch.stack([reward],dim=-1).detach().cpu().numpy()
            
            for k in mini_data.keys():
                mini_data[k] = np.stack(mini_data[k])
            info['mini_data'] = mini_data
            
            for k in collect_data.keys():
                collect_data[k] = np.stack(collect_data[k])
            info['collect_data'] = collect_data
        
        return obs, reward, done, bad_done, exceed_time_limit, info
