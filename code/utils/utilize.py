import matplotlib.pyplot as plt
import numpy as np
from sumolib import checkBinary
import os
import sys
import pickle
import xml.etree.cElementTree as ET
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import sys
sys.path.append("../network")
from utils.static_utilize import import_train_static_configuration


def import_train_configuration():
    """
    Read the config file regarding the training and import its content
    """

    config = {}

    # [test purpose]
    config['test_purpose'] = 'Continue to train with less penalty'
    config['network'] = 'Grid' # 'Grid' 'Bloomsbury'

    # [Perimeter mode]
    ''' DQN and Static: Must be centralize
        DDPG: Can be centralized & decentralized
    '''
    config['mode'] = 'train' # 'test' 'train'
    config['upper_mode'] = 'PI'  # 'Static' # 'DDPG' #'DQN' # 'Expert' # 'MaxPressure' # 'C_DQN' # 'PI' # 'FixTime'
    config['lower_mode'] = 'MaxPressure'  # 'FixTime'  #'OAM' # 'MaxPressure'
    config['peri_action_mode'] = 'centralize' # 'decentralize' 'centralize'
    config['peri_green_start_model'] = True     # 模型中是否考虑绿灯启亮时间对实际流入率的影响
    config['peri_spillover_penalty'] = False     # 模型中是否考虑溢出惩罚项
    config['peri_distribution_mode'] = 'balance_queue'  # 'equal' #  'balance_queue'
    config['peri_signal_phase_mode'] = 'Unfixed'  # 'NEMA' # 'Unfixed' # 'Slot'
    config['peri_optimization_mode'] = 'decentralize'  # 'centralize' # 'decentralize'

    # [state]
    if config['network'] == 'Grid':
        config['states'] = [
            'accu', 'accu_buffer', 'future_demand', 'network_mean_speed',     ## general
            'network_halting_vehicles', 'buffer_halting_vehicles',             ## halting
            # 'down_edge_occupancy','buffer_aver_occupancy'                      ## buffer specific
            ]
    if config['network'] == 'Bloomsbury':
        config['states'] = [
            'accu', 'network_mean_speed', 'future_demand',      ## general
            'network_halting_vehicles']
    # 'accu', accu_buffer,'future_demand', 'entered_vehs','network_mean_speed', 'network_halting_vehicles', 'buffer_halting_vehicles'
    #  'down_edge_occupancy','buffer_aver_occupancy', 

    # [normalization]
    if True:
        config['reward_max'] = 750
        config['entered_veh_max'] = 300
        config['accu_max'] = 3500 if config['network'] == 'Grid' else 6000
        config['accu_buffer_max'] = 3500
        config['max_queue'] = 200
        config['Demand_state_max'] = 2000 if config['network'] == 'Grid' else 5500
        config['network_mean_speed_max'] = 15
        config['PN_halt_vehs_max'] = 3000 if config['network'] == 'Grid' else 5000
        config['buffer_halt_vehs_max'] = 3000
        config['production_control_interval_max'] = 3000
        config['lower_reward_max'] = 2000


    # [simulation]
    if True:    
        # -1 allcores ## 0 single process ## 1+ multi-process
        config['n_jobs'] = 0
        config['expert_episode'] = 0
        config['gui'] = True  # True  # False  #
        config['control_interval'] = 20# 10 sec for lower level
        config['max_steps'] = 3000  # 10800 # 1000 # 6000
        # config['green_duration'] = green_duration
        config['min_green'] = 5.
        config['yellow_duration'] = 5.
        config['cycle_time'] = 100
        config['max_green_duration'] = 50
        config['max_green'] = 50
        # config['max_green_duration'] = 80
        # config['max_green'] = config[
        #     'cycle_time'] - 2 * config['yellow_duration'] - config['min_green']
        config['flow_rate'] = 0.9  # secs per vehicles
        # config['act_range'] = config['max_green'] // config['flow_rate'] * config[
        #     'EdgeCross'].shape[0]
        config['splitmode'] = 1  # 0 = waittime, 1 = waitveh
        config['slot_merge_cycle_num'] = 2      # Number of cycles in the generalized cycle in slot-based control
        config['slot_number'] = 7
        if config['peri_signal_phase_mode'] == 'Slot':
            config['cycle_time'] = config['cycle_time'] * config['slot_merge_cycle_num']  # update and record info
        config['infostep'] = config['cycle_time']

    # [Signal Optimization]
    if True:
        config['left_sfr'] = 1600/3600   # veh/s
        config['through_sfr'] = 1800/3600
        config['right_sfr'] = 1600/3600
        # config['saturation_flow_rate'] = {}
        config['saturation_flow_rate'] = 1550/3600
        config['saturation_limit'] = 1
        config['spillover_threshold'] = 0.9       # 判断排队溢出的阈值
        config['upper_obj_weight'] = {'gating': 1e6, 'balance': 1e2, 'spillover': 1e5}
        config['flow_weight'] = {'inflow': 1, 'outflow': 3, 'normal flow': 1}
        config['avg_spacing'] = 9
        config['max_iteration_step'] = 10

    # [PI controller setting]
    if config['upper_mode'] == 'PI':
        if config['network'] == 'Grid':
            config['accu_critic'] = 700    # 700 for 3*3
            config['K_p'] = 20 / 3600 * config['cycle_time']     # normalization
            config['K_i'] = 5 / 3600 * config['cycle_time']

        elif config['network'] == 'Bloomsbury':
            config['accu_critic'] = 700
            config['K_p'] = 20
            config['K_i'] = 20

    # [demand]
    if True:
        config['DemandConfig'] = [
            {
                'DemandType': 'Outside-PN (high)',     # 从西向东和从北向南为高流量方向
                'FromRegion': [25, 26],
                'ToRegion': [6, 7, 8, 11, 12, 13, 16, 17, 18],
                'VolumnRange': [[2000, 2000], [2000, 3000], [3000, 0], [0, 0]],
                # 'Demand_interval': [20, 20, 15, 5],
                'Demand_interval': [0.15, 0.2, 0.15, 0.5],     # 变成比例的形式, 和可能不为1
                'Multiplier': 1.8  # 0.1 #0.5 #1.0
            },
            {
                'DemandType': 'Outside-PN (low)',     # 从东向西和从南向北为低流量方向
                'FromRegion': [27, 28],
                'ToRegion': [6, 7, 8, 11, 12, 13, 16, 17, 18],
                'VolumnRange': [[2000, 2000], [2000, 3000], [3000, 0], [0, 0]],
                'Demand_interval': [0.15, 0.2, 0.15, 0.5],
                'Multiplier': 1  # 0.1 #0.5 #1.0
            },
            {
                'DemandType': 'PN-PN',
                'FromRegion': [6, 7, 8, 11, 12, 13, 16, 17, 18],
                'ToRegion': [6, 7, 8, 11, 12, 13, 16, 17, 18],
                'VolumnRange': [[400, 400], [400, 800], [800, 0], [0, 0]],
                'Demand_interval': [0.4, 0.2, 0.2, 0.2],
                'Multiplier': 1 #1.5  # 0.1 #0.5 #1
            },
            {
                'DemandType': 'PN-Outside (high)',     # 从西向东和从北向南为高流量方向
                'FromRegion': [6, 7, 8, 11, 12, 13, 16, 17, 18],
                'ToRegion': [27, 28],
                'VolumnRange': [[2000, 2000], [2000, 2500], [2500, 0], [0, 0]],
                'Demand_interval': [0.15, 0.2, 0.15, 0.5],
                'Multiplier': 1.5 #1.5  # 0.1 #0.5 #1
            },
            {
                'DemandType': 'PN-Outside (low)',
                'FromRegion': [6, 7, 8, 11, 12, 13, 16, 17, 18],
                'ToRegion': [25, 26],
                'VolumnRange': [[2000, 2000], [2000, 2500], [2500, 0], [0, 0]],
                'Demand_interval': [0.15, 0.2, 0.15, 0.5],
                'Multiplier': 1  # 1.5  # 0.1 #0.5 #1
            }
        ]
        config['DemandNoise'] = {'noise_mean': 0, 'noise_variance': 0.1}
        config['Demand_interval_sec'] = 100  # 100sec for each interval
        config['Demand_interval_total'] = int(config['max_steps'] / config['Demand_interval_sec'])
        # 修改Demand_interval
        for demand in config['DemandConfig']:
            demand['Demand_interval'] = [round(ratio * config['Demand_interval_total']) for ratio in demand['Demand_interval']]
            interval_error = config['Demand_interval_total'] - sum(demand['Demand_interval'])
            demand['Demand_interval'][-1] += interval_error
        # future steps of demand in the states
        config['Demand_state_steps'] = 3

    # [agent_upper]
    if True:
        config['reward_delay_steps'] = 0
        config['penalty_delay_steps'] = 0
        config['multi-step'] = 1
        config['rl_type'] = 'off-policy'  # 'off-policy' 'on-policy'

        if config['rl_type'] == 'off-policy':  # off-policy
            if config['multi-step'] == 1:  # real off-policy
                config['buffer_size'] = int(5000)
            else:  # relatively shorter
                config['buffer_size'] = int(5000)
        else:  # on-policy
            config['buffer_size'] = int(108 * max([config['n_jobs'], 1]))

        config['penalty_type'] = 'None'  # 'queue'  # 'delta_queue'
        config['epsilon'] = 0.9
        config['explore_decay'] = 0.9
        config['explore_lb'] = 0.01
        config['reward_normalization'] = False  # True
        config['sample_mode'] = 'random'  # balance # random
        config['state_steps'] = 1  # number of stacked steps for the states
        config['gamma_multi_step'] = 0.9
        config['gamma'] = 0.9
        config['lr_C'] = 5e-4
        config['lr_C_decay'] =.98
        config['lr_C_lb'] = 1e-4

        config['lr_A'] = 1e-4
        config['tau'] = 1e-2
        config['batch_size'] = 128 #64  # 128
        config['total_episodes'] = 10 #1000 # 500
        config['online'] = False  # True #False
        # config['input_dim'] = len(config['states']) * config['state_steps']
        if config['peri_action_mode'] == 'centralize':
            config['act_dim'] = 1
        else:
            config['act_dim'] = len(config['Peri_info'])

        # config['act_range'] = config['max_green'] * len(config['Peri_info'])
        config['reuse_time'] = 2
        # config['action_interval'] = np.array(action_interval)
        # config['initial_std'] = 0.9
        config['target_update_freq'] = 5  # >=1
    
    # [agent_lower]
    if True:
        config['lr_C_lower'] = 1e-4
        config['epsilon_lower'] = 0.99
        config['explore_decay_lower'] = 0.8
        config['buffer_size_lower'] = 6e4
        config['gamma_lower'] = 0.85
        config['reuse_time_lower'] = 1

    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the svisual mode
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [
        sumoBinary,
        "-c",
        os.path.join(sumocfg_file_name),
        "--no-step-log", "true",
        "--verbose", "true",
        "--no-warnings","true",
        # "--no-internal-links","true",
        # "--waiting-time-memory",
        # str(max_steps),
        # "--error-log","network/GridBuffer/error.txt"
        "--time-to-teleport", "120",
        "--queue-output", config['queuefile_dir']
    ]

    return sumo_cmd


############### Path ###################
def set_train_path(path_name, type):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    path_name = os.path.join(os.getcwd(), path_name, '')
    os.makedirs(os.path.dirname(path_name), exist_ok=True)

    dir_content = os.listdir(path_name)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    path_name = os.path.join(path_name, f'{type}_' + new_version, '')
    os.makedirs(os.path.dirname(path_name), exist_ok=True)

    if type == 'plot':
        ## test
        path_name_test = os.path.join(path_name, 'test', '')
        os.makedirs(os.path.dirname(path_name_test), exist_ok=True)
        
        ## critic
        path_name_critic = os.path.join(path_name, 'critic', '')
        os.makedirs(os.path.dirname(path_name_critic), exist_ok=True)

        ## metric
        path_name_metric = os.path.join(path_name, 'metric', '')
        os.makedirs(os.path.dirname(path_name_metric), exist_ok=True)

        ## explore
        path_name_explore = os.path.join(path_name, 'explore', '')
        os.makedirs(os.path.dirname(path_name_explore), exist_ok=True)
        
        ## model
        path_name_model = os.path.join(path_name, 'model', '')
        os.makedirs(os.path.dirname(path_name_model), exist_ok=True)

        return path_name, path_name_model

    else:
        return path_name


def set_test_path(path_name):
    ''' set the path to load the well-trained model
    '''
    path_name = os.path.join(os.getcwd(), path_name, '')
    return path_name


def write_log(config):
    with open(f"{config['plots_path_name']}A-readme.txt", "a") as file:
        print(f"PURPOSE = {config['test_purpose']}\n", file=file)
        print(f"Mode = {config['mode']}\n", file=file)
        print(f"Upper_Mode = {config['upper_mode']}\n", file=file)
        print(f"Lower_Mode = {config['lower_mode']}\n", file=file)
        print(f"RL-TYPE = {config['rl_type']}\n", file=file)
        print(f"STATE = {config['states']}\n", file=file)
        print(f"Multi-step = {config['multi-step']}\n", file=file)
        print(f"Expert episodes = {config['expert_episode']}\n", file=file)
        print(f"Buffer size = {config['buffer_size']}\n", file=file)
        print(f"Sample_mode = {config['sample_mode']}\n", file=file)
        print(f"Penalty_type = {config['penalty_type']}\n", file=file)
        # print(f"STACKED_STEPS = {config['state_steps']}\n", file=file)
        print(f"DEMAND = {config['DemandConfig']}\n", file=file)
        print(f"Gamma = {config['gamma']}\n", file=file)
        print(f"Gamma_multi_step = {config['gamma_multi_step']}\n", file=file)
        print(f"LR_CRITIC = {config['lr_C']}\n", file=file)
        print(f"LR_CRITIC_DECAY = {config['lr_C_decay']}\n", file=file)
        print(f"LR_ACTOR = {config['lr_A']}\n", file=file)
        print(f"TAU = {config['tau']}\n", file=file)
        print(f"Reuse time = {config['reuse_time']}\n", file=file)
        print(f"Epsilon = {config['epsilon']}\n", file=file)
        print(f"Epsilon_decay = {config['explore_decay']}\n", file=file)
        print(
            f"CREDIT ASSIGHMENT_reward_delay_steps= {config['reward_delay_steps']}\n", file=file)
        print(
            f"CREDIT ASSIGHMENT_penalty_delay_steps= {config['penalty_delay_steps']}\n", file=file)
        print(
            f"Target network update frequency= {config['target_update_freq']}\n", file=file)
        print(
            f"Reward normalization in batch replay= {config['reward_normalization']}\n", file=file)


def save_config(config):
    # np.save(f"{config['models_path_name']}config.npy", config)
    # 每次运行的config与statistics保存在一起
    filename = 'config.pkl'
    stats_path = os.path.join(config['stats_path_name'], filename)
    with open(stats_path, 'wb') as f:
        pickle.dump(config, f)


class Test:
    def __init__(self):
        self.obj_list = []
        self.reward_list = []
        self.penalty_list = []
        self.accu_list = []
        self.throu_list = []
        self.action_list = []
    
    def record_data(self,cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions):
        self.obj_list.append(cumul_obj)
        self.reward_list.append(cumul_reward)
        self.penalty_list.append(cumul_penalty)
        self.accu_list.append(accu_episode)
        self.throu_list.append(throughput_episode)
        self.action_list.append(actions)

    def save_data_test(self):
        ''' save objective, reward, penalty, throughput, actions along the testing process  
        '''
        data_test = {}

        data_test['obj_list'] = self.obj_list
        data_test['reward_list'] = self.reward_list
        data_test['penalty_epis'] = self.penalty_list
        data_test['accu_list'] = self.accu_list
        data_test['throu_list'] = self.throu_list
        data_test['action_list'] = self.action_list

        save_dir = config['models_path_name'] + 'data_test.pkl'
        with open(save_dir, 'wb') as f:
            pickle.dump([data_test], f)

        print('###### Data save: Success ######')
        

config = import_train_configuration()
config = import_train_static_configuration(config)


if __name__ == "__main__":
    save_config(config)
    write_log(config)
