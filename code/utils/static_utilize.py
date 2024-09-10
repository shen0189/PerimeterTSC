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


def import_train_static_configuration(config):
    """
    Add the static config to the changeable config file
    """

    # [network_config]
    if config['network'] == 'Grid':
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
            'NodePN': [13, 12, 22, 21, 20, 19, 18, 17, 34],
            'NodePeri': [50, 51, 52, 53, 54, 55]
        }
        # read edge data
        df = pd.read_csv("network\Bloomsbury\edge_bloom.csv", sep=';', usecols=['edge_id', 'edge_PN'])

        # config['Edge'] = sorted(df['edge_id'].to_numpy)
        config['Edge'] = sorted(df['edge_id'].values.tolist())
        config['Edge_PN'] = sorted(df[df['edge_PN'] == 1]['edge_id'].values.tolist())
        config['Edge_Peri'] = sorted(df[df['edge_PN'] == 0]['edge_id'].values.tolist())

    # [perimeter signal_config]
    if config['network'] == 'Grid':  # 'Grid':
        # config['EdgeCross'] = np.array([92, 93, 94, 95])
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
                'inflow_lane_groups': [3],   # 列表长度代表车道组数量, 第i个元素代表车道组i的车道数量, 从最外侧开始数
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
                'inflow_lane_groups': [3],
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
                'inflow_lane_groups': [3],
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
                'inflow_lane_groups': [3],
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
                'buffer_edges': [1050, 1046, 1048],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '3': {
                'edge': 1202,
                'down_edge': 1010,
                'buffer_edges': [1010, 1036, 1035, 1034],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '8': {
                'edge': 1205,
                'down_edge': 1023,
                'buffer_edges': [1023, 1027, 1028],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '32': {
                'edge': 1207,
                'down_edge': 1113,
                'buffer_edges': [1113, 1117, 1115],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '43': {
                'edge': 1209,
                'down_edge': 1150,
                'buffer_edges': [1150, 1123, 1124, 1125],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
            '39': {
                'edge': 1211,
                'down_edge': 1140,
                'buffer_edges': [1136, 1135, 1136],
                'phase_info':
                    ['control_phase', 'yellow_phase', '', 'yellow_phase']
            },
        }
    config['inflow_edge'] = [signal_info['edge'] for signal_info in config['Peri_info'].values()]

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
        config['tripfile_dir'] = "measurements/GridBufferNew/trip.xml"

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


# class Test:
#     def __init__(self):
#         self.obj_list = []
#         self.reward_list = []
#         self.penalty_list = []
#         self.accu_list = []
#         self.throu_list = []
#         self.action_list = []
#
#     def record_data(self, cumul_obj, cumul_reward, cumul_penalty, accu_episode, throughput_episode, actions):
#         self.obj_list.append(cumul_obj)
#         self.reward_list.append(cumul_reward)
#         self.penalty_list.append(cumul_penalty)
#         self.accu_list.append(accu_episode)
#         self.throu_list.append(throughput_episode)
#         self.action_list.append(actions)
#
#     def save_data_test(self):
#         ''' save objective, reward, penalty, throughput, actions along the testing process
#         '''
#         data_test = {}
#
#         data_test['obj_list'] = self.obj_list
#         data_test['reward_list'] = self.reward_list
#         data_test['penalty_epis'] = self.penalty_list
#         data_test['accu_list'] = self.accu_list
#         data_test['throu_list'] = self.throu_list
#         data_test['action_list'] = self.action_list
#
#         save_dir = config['models_path_name'] + 'data_test.pkl'
#         with open(save_dir, 'wb') as f:
#             pickle.dump([data_test], f)
#
#         print('###### Data save: Success ######')



