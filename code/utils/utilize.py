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
    config['upper_mode'] = 'PI'  # 'Static' # 'DDPG' #'DQN' # 'Expert' # 'MaxPressure' # 'C_DQN' # 'PI'
    config['lower_mode'] = 'MaxPressure'  # 'FixTime'  #'OAM' # 'MaxPressure'
    config['peri_action_mode'] = 'centralize' # 'decentralize' 'centralize'
    config['peri_green_start_model'] = True     # 模型中是否考虑绿灯启亮时间对实际流入率的影响
    config['peri_spillover_penalty'] = True     # 模型中是否考虑溢出惩罚项
    config['peri_distribution_mode'] = 'equal'  # 'equal' #  'balance_queue'
    config['peri_signal_phase_mode'] = 'NEMA'  # 'NEMA' # 'Unfixed' # 'Slot'
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

    # [network_config]
    if config['network'] == 'Grid':
        # config['Node'] = {
        #     'NodePN': [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24],
        #     'NodePeri': [29, 30, 31, 32]
        # }
        # config['Edge'] = list(range(0, 96))
        # config['Edge_Peri'] = sorted([80, 92, 84, 93, 87, 94, 89, 95])
        # config['Edge_PN'] = sorted([
        #     9, 11, 6, 8, 5, 3, 0, 2, 29, 26, 17, 14, 47, 44, 35, 32, 65, 62, 53, 50, 79, 77, 76, 74, 73, 71, 68, 70,
        #     67, 51, 49, 33, 31, 15, 13, 1, 69, 55, 16, 4, 20, 7, 72, 59, 63, 75, 24, 10, 28, 12, 46, 30, 64, 48, 78, 66,
        #     18, 19, 21, 22, 23, 25, 27, 34, 36, 37, 39, 38, 40, 41, 43, 42, 45, 52,
        #     54, 57, 56, 58, 60, 61
        # ])
        # config['Edge_PN_out'] = sorted([81, 85, 88, 87])  # the outflows of PN
        config['Node'] = {
            'NodePN': [6, 7, 8, 11, 12, 13, 16, 17, 18],
            'NodePeri': [25, 26, 27, 28]
        }
        config['Edge'] = list(range(0, 88))
        config['Edge_Peri'] = list(range(80, 88))
        config['Edge_PN'] = sorted([
            18, 19, 21, 22, 23, 25, 27, 34, 36, 37, 39, 38, 40, 41, 43, 42, 45, 52,
            54, 57, 56, 58, 60, 61
        ])
        config['Edge_PN_out'] = sorted([16, 17, 20, 24, 26, 44, 62, 63, 59, 55, 53,
                                        35])  # the outflows of PN
    elif config['network'] == 'Bloomsbury':
        config['Node'] = {
            # 'NodePN': [ 14,13,12,11,9,22,21,20,19,18,17,33,34,35,38],
            'NodePN': [13, 12, 22, 21, 20, 19, 18, 17, 34],
            'NodePeri': [50, 51, 52, 53, 54, 55]
            # 'NodePeri': [50]
        }
        # read edge data
        df = pd.read_csv("network\Bloomsbury\edge_bloom.csv", sep=';', usecols=['edge_id','edge_PN'])
        
        # config['Edge'] = sorted(df['edge_id'].to_numpy)
        config['Edge'] = sorted(df['edge_id'].values.tolist())
        config['Edge_PN'] = sorted(df[df['edge_PN'] == 1]['edge_id'].values.tolist())
        config['Edge_Peri'] = sorted(df[df['edge_PN'] == 0]['edge_id'].values.tolist())


    # [perimeter signal_config]
    if config['network'] == 'Grid': # 'Grid':
        # config['EdgeCross'] = np.array([92, 93, 94, 95])
        # Peri_info: All perimeter movements
        config['Peri_info'] = {
            'A2': {
                'node': '2',
                'edge': 80,
                'in_edges': [8, 20, 3, 80],  # 按照sumo中定义phase state的顺序 从北进口开始顺时针方向
                'down_edge': 7,
                'buffer_edges': [5, 6, 7],
                'external_in_edges': [80],
                'external_out_edges': [81],
                'internal_in_edges': [8, 20, 3],
                'internal_out_edges': [5, 6, 7],
                'nema_plan': {'ring1': ['3_s', '8_l', '', '20_s'],
                              'ring2': ['8_s', '3_l', '', '80_s', '20_l']},
                'phase_info':
                    ['', 'yellow_phase', '', 'yellow_phase', 'control_phase', 'yellow_phase', 'control_phase',
                     'yellow_phase'],  # 根据group-based优化结果修改
                'slot_stage_plan': {'stage1': ('3_l', '3_s'), 'stage2': ('3_l', '8_l'),
                                    'stage3': ('8_l', '8_s'), 'stage4': ('3_s', '8_s'),
                                    'stage5': ('20_l',), 'stage6': ('80_s',), 'stage7': ('20_s', '20_l'),
                                    'stage8': ('20_s', '80_s')}
            },
            'C4': {
                'node': '14',
                'edge': 82,
                'in_edges': [82, 64, 44, 30],
                'down_edge': 47,
                'buffer_edges': [48, 47, 46],
                'external_in_edges': [82],
                'external_out_edges': [83],
                'internal_in_edges': [30, 44, 64],
                'internal_out_edges': [48, 47, 46],
                'nema_plan': {'ring1': ['82_s', '44_l', '', '64_s', '30_l'],
                              'ring2': ['44_s', '', '30_s', '64_l']},
                'phase_info': ['control_phase', 'yellow_phase', 'control_phase', 'yellow_phase',
                               '', 'yellow_phase', '', 'yellow_phase'],
                'slot_stage_plan': {'stage1': ('30_l', '30_s'), 'stage2': ('30_l', '64_l'),
                                    'stage3': ('64_l', '64_s'), 'stage4': ('30_s', '64_s'),
                                    'stage5': ('44_l',), 'stage6': ('82_s',), 'stage7': ('44_s', '44_l'),
                                    'stage8': ('44_s', '82_s')}
            },
            'E2': {
                'node': '22',
                'edge': 84,
                'in_edges': [76, 84, 71, 59],
                'down_edge': 72,
                'buffer_edges': [73, 72, 74],
                'external_in_edges': [84],
                'external_out_edges': [85],
                'internal_in_edges': [59, 71, 76],
                'internal_out_edges': [73, 72, 74],
                'nema_plan': {'ring1': ['76_s', '71_l', '', '84_s', '59_l'],
                              'ring2': ['71_s', '76_l', '', '59_s']},
                'phase_info': ['', 'yellow_phase', '', 'yellow_phase',
                               'control_phase', 'yellow_phase', 'control_phase', 'yellow_phase'],
                'slot_stage_plan': {'stage1': ('71_l', '71_s'), 'stage2': ('71_l', '76_l'),
                                    'stage3': ('76_l', '76_s'), 'stage4': ('71_s', '76_s'),
                                    'stage5': ('59_l',), 'stage6': ('84_s',), 'stage7': ('59_s', '59_l'),
                                    'stage8': ('59_s', '84_s')}
            },
            'C0': {
                'node': '10',
                'edge': 86,
                'in_edges': [35, 49, 86, 15],
                'down_edge': 32,
                'buffer_edges': [31, 32, 33],
                'external_in_edges': [86],
                'external_out_edges': [87],
                'internal_in_edges': [15, 35, 49],
                'internal_out_edges': [31, 32, 33],
                'nema_plan': {'ring1': ['35_s', '', '49_s', '15_l'],
                              'ring2': ['86_s', '35_l', '', '15_s', '49_l']},
                'phase_info': ['control_phase', 'yellow_phase', 'control_phase', 'yellow_phase',
                               '', 'yellow_phase', '', 'yellow_phase'],
                'slot_stage_plan': {'stage1': ('15_l', '15_s'), 'stage2': ('15_l', '49_l'),
                                    'stage3': ('49_l', '49_s'), 'stage4': ('15_s', '49_s'),
                                    'stage5': ('35_l',), 'stage6': ('86_s',), 'stage7': ('35_s', '35_l'),
                                    'stage8': ('35_s', '86_s')}
            }
        }
    elif config['network'] == 'Bloomsbury': 
        config['Peri_info'] = {
            '15': {
                'edge': 1201,
                'down_edge': 1050,
                'buffer_edges': [1050,1046, 1048],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '3': {
                'edge': 1202,
                'down_edge': 1010,
                'buffer_edges': [1010,1036,1035,1034],
                'phase_info':
                ['control_phase', 'yellow_phase','', 'yellow_phase']
            },
            '8': {
                'edge': 1205,
                'down_edge': 1023,
                'buffer_edges': [1023,1027,1028],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '32': {
                'edge': 1207,
                'down_edge': 1113,
                'buffer_edges': [1113,1117,1115],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '43': {
                'edge': 1209,
                'down_edge': 1150,
                'buffer_edges': [1150,1123,1124,1125],
                'phase_info':
                [ 'control_phase', 'yellow_phase','', 'yellow_phase']
            },
            '39': {
                'edge': 1211,
                'down_edge': 1140,
                'buffer_edges': [1136,1135,1136],
                'phase_info':
                ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
        }
    config['inflow_edge'] = [signal_info['edge'] for signal_info in config['Peri_info'].values()]

    # [simulation]
    if True:    
        # -1 allcores ## 0 single process ## 1+ multi-process
        config['n_jobs'] = 0
        config['expert_episode'] = 0
        config['gui'] = True  # True  #False  #
        config['control_interval'] = 20# 10 sec for lower level
        config['max_steps'] = 6000  # 10800 # 1000 # 6000
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
        config['spillover_threshold'] = 1       # 判断排队溢出的阈值
        config['obj_weight'] = {'gating': 1e6, 'balance': 1e2, 'local': 1}
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


    # [dir]
    config['savefile_dir'] = "output/"
    config['models_path_name'] = "output/model"
    config['plots_path_name'] = 'output/plots'
    config['cache_path_name'] = 'output/cache'
    config['stats_path_name'] = 'output/statistics'


    if config['network'] == 'Grid':
        # config['sumocfg_file_name'] = 'network/GridBuffer/GridBuffer.sumocfg'
        # config['edgefile_dir'] = "network/GridBuffer/GridBuffer.edg.xml"
        # config['netfile_dir'] = "network/GridBuffer/GridBuffer.net.xml"
        # config['routefile_dir'] = "network/GridBuffer/GridBuffer.rou.xml"
        # config['edge_outputfile_dir'] = "measurements/GridBuffer/EdgeMesurements.xml"
        # config['lane_outputfile_dir'] = "measurements/GridBuffer/EdgeMesurements_lower.xml"
        # config['queuefile_dir'] = "measurements/GridBuffer/queue.xml"

        config['sumocfg_file_name'] = 'network/GridBufferNew/GridBuffer.sumocfg'
        config['edgefile_dir'] = "network/GridBufferNew/GridBuffer.edg.xml"
        config['netfile_dir'] = "network/GridBufferNew/GridBuffer.net.xml"
        config['routefile_dir'] = "network/GridBufferNew/GridBuffer.rou.xml"
        config['edge_outputfile_dir'] = "measurements/GridBufferNew/EdgeMesurements.xml"
        config['lane_outputfile_dir'] = "measurements/GridBufferNew/EdgeMesurements_lower.xml"
        config['queuefile_dir'] = "measurements/GridBufferNew/queue.xml"
        
        # tls configure 
        if config['lower_mode'] == 'OAM':  # 'FixTime'  #'OAM' # 'MaxPressure'
            config['tls_config_name'] = './code/tls_new1.pkl'
        elif config['lower_mode'] == 'MaxPressure':
            config['tls_config_name'] = './code/tls_new07.pkl'
    
    if config['network'] == 'Bloomsbury':
        config['sumocfg_file_name'] = 'network/Bloomsbury/Bloomsbury.sumocfg'
        config['netfile_dir'] = "network/Bloomsbury/Bloomsbury.net.xml"
        config['edgefile_dir'] = "network/Bloomsbury/Bloomsbury.edg.xml"
        config['routefile_dir'] = "network/Bloomsbury/Bloomsbury.rou.xml"
        config['queuefile_dir'] = "measurements/Bloomsbury/queue.xml"
        config['edge_outputfile_dir'] = "measurements/Bloomsbury/EdgeMesurements.xml"
        config['lane_outputfile_dir'] = "measurements/Bloomsbury/EdgeMesurements_lower.xml"

                # tls configure 
        if config['lower_mode'] == 'OAM':  # 'FixTime'  #'OAM' # 'MaxPressure'
            config['tls_config_name'] = './code/tls_bloom_1.pkl'
        elif config['lower_mode'] == 'MaxPressure':
            config['tls_config_name'] = './code/tls_bloom_05.pkl'




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

def get_NodeConfig(edgefile_dir):
    ''' read the edge file 'edg.xml', obtain the indege and outedge of each node
    '''
    tree = ET.ElementTree(file=edgefile_dir)
    root = tree.getroot()
    NodeConfig = {}
    for child in root:
        edge_id = int(child.attrib['id'])
        From_Node = int(child.attrib['from'])
        To_Node = int(child.attrib['to'])

        # add From_Node
        if From_Node not in NodeConfig:
            NodeConfig[From_Node] = {'InEdge': [], 'OutEdge': []}

        NodeConfig[From_Node]['OutEdge'].append(edge_id)

        # add From_Node
        if To_Node not in NodeConfig:
            NodeConfig[To_Node] = {'InEdge': [], 'OutEdge': []}

        NodeConfig[To_Node]['InEdge'].append(edge_id)

    return NodeConfig

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

if __name__ == "__main__":
    save_config(config)
    write_log(config)
